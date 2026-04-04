#!/usr/bin/env python3
"""
只测一次 prefill（整段序列、无 decode）。可选与标准 HuggingFace Qwen3 同输入对比，便于在 Perfetto 里看「多出来的是谁」。

用法（在 Lazy-Llama 目录下）:
  export CUDA_VISIBLE_DEVICES=0
  python profile_prefill.py --seq-len 4096 --runs 5
  python profile_prefill.py --seq-len 8192 --profile --trace-dir ./traces --compare-baseline

对比时：
  - 同 --seed、同 input_ids / attention_mask；Baseline 使用与 Lazy 相同的 --attn-impl（默认 sdpa）。
  - 先卸载 Lazy 再加载 Baseline，避免 24G 卡上双份 8B OOM。
  - 生成两份 trace 目录：.../lazy_prefill_seq{N} 与 .../baseline_prefill_seq{N}，用 Perfetto 分别打开对照时间轴。

说明:
  - 默认 --max-seq-len 0 → seq_len+512，避免 35200 KV/Aux 撑爆显存。
  - 默认 --attn-impl sdpa；eager 长序列会物化 L×L，易 OOM。
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import torch
from transformers import AutoConfig, Qwen3ForCausalLM

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from caches import AuxCache, KVCache
from models import LazyQwen3ForCausalLM


def build_lazy_model(
    local_path: str,
    device: torch.device,
    pruning_rate: float,
    num_layers: int,
) -> LazyQwen3ForCausalLM:
    qwen3 = Qwen3ForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    cfg = qwen3.config
    state_dict = qwen3.state_dict()
    del qwen3
    gc.collect()
    rates = {i: float(pruning_rate) for i in range(num_layers)}
    model = LazyQwen3ForCausalLM.from_qwen3_state_dict(state_dict, cfg, pruning_rates=rates)
    del state_dict
    gc.collect()
    model = model.to(device)
    model.eval()
    return model


def build_prefill_batch(
    *,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    hi = min(vocab_size, 32000)
    batch_size = 1
    input_ids = torch.randint(
        low=0,
        high=hi,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
        generator=g,
    )
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
    }


def run_lazy_prefill_once(
    model: LazyQwen3ForCausalLM,
    *,
    batch: dict[str, Any],
    kv: KVCache,
    aux: AuxCache,
) -> None:
    kv.reset()
    aux.reset()
    with torch.no_grad():
        buf = StringIO()
        with redirect_stdout(buf):
            model.model(
                kv_cache=kv,
                aux_cache=aux,
                cache_position=batch["cache_position"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                output_attentions=False,
            )


def run_baseline_prefill_once(model: Qwen3ForCausalLM, batch: dict[str, Any]) -> None:
    with torch.no_grad():
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


def _release_cuda(*objs: Any) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _profile_one_lazy(
    *,
    args: argparse.Namespace,
    device: torch.device,
    batch: dict[str, Any],
    max_seq_cap: int,
    num_layers: int,
) -> None:
    model = build_lazy_model(
        args.local_path,
        device,
        pruning_rate=args.pruning_rate,
        num_layers=num_layers,
    )
    kv = KVCache(
        model.config.num_hidden_layers,
        1,
        model.config.num_key_value_heads,
        max_seq_cap,
        model.config.head_dim,
        device,
        dtype=torch.bfloat16,
    )
    aux = AuxCache(
        model.config.num_hidden_layers,
        1,
        max_seq_cap,
        model.config.hidden_size,
        device,
        dtype=torch.bfloat16,
    )
    for _ in range(args.warmup):
        run_lazy_prefill_once(model, batch=batch, kv=kv, aux=aux)
    torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    trace_path = os.path.join(args.trace_dir, f"lazy_prefill_seq{args.seq_len}")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
    ) as prof:
        with torch.no_grad():
            buf = StringIO()
            with redirect_stdout(buf):
                run_lazy_prefill_once(model, batch=batch, kv=kv, aux=aux)
    torch.cuda.synchronize()
    print(f"[profile] Lazy trace: {trace_path}")
    if args.profile_table:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=int(args.profile_table_rows)))

    _release_cuda(model, kv, aux)


def _profile_one_baseline(
    *,
    args: argparse.Namespace,
    device: torch.device,
    batch: dict[str, Any],
) -> None:
    model = Qwen3ForCausalLM.from_pretrained(
        args.local_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.config._attn_implementation = args.attn_impl
    model = model.to(device)
    model.eval()

    for _ in range(args.warmup):
        run_baseline_prefill_once(model, batch)
    torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    trace_path = os.path.join(args.trace_dir, f"baseline_prefill_seq{args.seq_len}")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
    ) as prof:
        with torch.no_grad():
            run_baseline_prefill_once(model, batch)
    torch.cuda.synchronize()
    print(f"[profile] Baseline trace: {trace_path}")
    if args.profile_table:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=int(args.profile_table_rows)))

    _release_cuda(model)


def main() -> None:
    p = argparse.ArgumentParser(description="LazyQwen3 prefill-only timing / profiler，可选 Baseline 对照")
    p.add_argument("--local-path", default=os.environ.get("QWEN3_LOCAL_PATH", "/data/zn/model/models/Qwen3-8B"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="KV/Aux 预分配上限；0=自动 seq_len+512。",
    )
    p.add_argument("--pruning-rate", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42, help="Lazy 与 Baseline 共用，保证同输入")
    p.add_argument("--compare-baseline", action="store_true", help="再跑标准 Qwen3 prefill（先释放 Lazy 再加载）")
    p.add_argument("--profile", action="store_true", help="导出 Chrome trace（可配合 --compare-baseline 出两份）")
    p.add_argument("--trace-dir", type=str, default="./profiler_traces")
    p.add_argument(
        "--profile-table",
        action="store_true",
        help="profile 结束后打印按 GPU 时间排序的算子表（便于快速对比）",
    )
    p.add_argument("--profile-table-rows", type=int, default=20)
    p.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=("sdpa", "eager", "flash_attention_2"),
        help="Lazy DecoderLayer 与 Baseline config 共用。",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["LAZYLLAMA_ATTN_IMPL"] = args.attn_impl

    cfg = AutoConfig.from_pretrained(args.local_path)
    num_layers = int(cfg.num_hidden_layers)
    vocab_size = int(cfg.vocab_size)

    headroom = 512
    max_seq_cap = args.max_seq_len if args.max_seq_len > 0 else (args.seq_len + headroom)
    if args.seq_len > max_seq_cap:
        raise SystemExit(f"--seq-len {args.seq_len} 不能大于有效 max_seq_len={max_seq_cap}")
    if args.max_seq_len <= 0:
        print(f"[cache] max_seq_len={max_seq_cap}（自动 seq_len+{headroom}）")

    batch = build_prefill_batch(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        device=device,
        seed=args.seed,
    )

    if args.profile:
        _profile_one_lazy(
            args=args,
            device=device,
            batch=batch,
            max_seq_cap=max_seq_cap,
            num_layers=num_layers,
        )
        if args.compare_baseline:
            _profile_one_baseline(args=args, device=device, batch=batch)
        print("[profile] Perfetto: https://ui.perfetto.dev 打开上述目录中的 .json")
        return

    # ----- timing only: Lazy -----
    model = build_lazy_model(
        args.local_path,
        device,
        pruning_rate=args.pruning_rate,
        num_layers=num_layers,
    )
    kv = KVCache(
        model.config.num_hidden_layers,
        1,
        model.config.num_key_value_heads,
        max_seq_cap,
        model.config.head_dim,
        device,
        dtype=torch.bfloat16,
    )
    aux = AuxCache(
        model.config.num_hidden_layers,
        1,
        max_seq_cap,
        model.config.hidden_size,
        device,
        dtype=torch.bfloat16,
    )
    for _ in range(args.warmup):
        run_lazy_prefill_once(model, batch=batch, kv=kv, aux=aux)
    torch.cuda.synchronize()

    lazy_ms: list[float] = []
    for _ in range(args.runs):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        with torch.no_grad():
            buf = StringIO()
            with redirect_stdout(buf):
                run_lazy_prefill_once(model, batch=batch, kv=kv, aux=aux)
        ender.record()
        torch.cuda.synchronize()
        lazy_ms.append(starter.elapsed_time(ender))

    lazy_avg = sum(lazy_ms) / len(lazy_ms)
    print(
        f"[prefill Lazy] seq_len={args.seq_len} runs={args.runs} "
        f"ms: {[round(t, 2) for t in lazy_ms]} avg={lazy_avg:.2f} ms"
    )

    if not args.compare_baseline:
        return

    _release_cuda(model, kv, aux)

    base = Qwen3ForCausalLM.from_pretrained(
        args.local_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    base.config._attn_implementation = args.attn_impl
    base = base.to(device)
    base.eval()
    for _ in range(args.warmup):
        run_baseline_prefill_once(base, batch)
    torch.cuda.synchronize()

    base_ms: list[float] = []
    for _ in range(args.runs):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        with torch.no_grad():
            run_baseline_prefill_once(base, batch)
        ender.record()
        torch.cuda.synchronize()
        base_ms.append(starter.elapsed_time(ender))

    base_avg = sum(base_ms) / len(base_ms)
    print(
        f"[prefill Baseline] seq_len={args.seq_len} runs={args.runs} "
        f"ms: {[round(t, 2) for t in base_ms]} avg={base_avg:.2f} ms"
    )
    print(f"[对比] Lazy avg / Baseline avg = {lazy_avg / base_avg:.3f}x")


if __name__ == "__main__":
    main()
