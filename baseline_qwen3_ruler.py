#!/usr/bin/env python3
"""
标准 HuggingFace Qwen3ForCausalLM 在 RULER 合成任务 jsonl 上推理，与 Lazy-Llama/example.py 的 run_ruler_jsonl 对齐：

- 数据：RULER prepare.py 生成的 jsonl（字段 index / input / outputs）
- batch_size=1，bfloat16，do_sample=False
- max_seq_len、tokenizer truncation（max_length = max_seq_len - max_gen）、
  max_new_tokens = min(max_gen, max_seq_len - input_len)，其中 max_gen 与 prepare 时 tokens_to_generate 一致（按任务目录名从 synthetic.yaml 推断）
- 记录每条 prefill_ms（generate 内首次 model.forward）与 total_generate_ms
- 支持 *_checkpoint.json 断点续跑（逻辑与 example.py 一致）

运行前设置 GPU，例如：
  export CUDA_VISIBLE_DEVICES=4
  python baseline_qwen3_qasper.py --ruler-jsonl /data/zn/RULER/data/niah_single_1/validation.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3ForCausalLM

RULER_ROOT = os.environ.get("RULER_ROOT", "/data/zn/RULER")

RULER_BASE_TASK_MAX_NEW_TOKENS = {
    "niah": 128,
    "variable_tracking": 30,
    "common_words_extraction": 120,
    "freq_words_extraction": 50,
    "qa": 32,
}

DEFAULT_LOCAL_PATH = "/data/zn/model/models/Qwen3-8B"
DEFAULT_MAX_SEQ_LEN = 35200
DEFAULT_RULER_SAVE_DIR = os.environ.get("RULER_SAVE_DIR", "/data/zn/ruler/4096")
DEFAULT_RULER_JSONL = os.path.join(DEFAULT_RULER_SAVE_DIR, "niah_single_1", "validation.jsonl")
DEFAULT_OUT = "/data/zn/Lazy-Llama/pred/baseline_qwen3/ruler_niah_single_1.jsonl"


def _infer_task_dir_name_from_jsonl(jsonl_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))


def _auto_output_path_for_jsonl(jsonl_path: str, *, output_default: str) -> str:
    task_name = _infer_task_dir_name_from_jsonl(jsonl_path)
    out_dir = os.path.dirname(os.path.abspath(output_default))
    return os.path.join(out_dir, f"ruler_{task_name}.jsonl")


def _load_ruler_synthetic_yaml() -> dict:
    yaml_path = os.path.join(RULER_ROOT, "scripts", "synthetic.yaml")
    if not os.path.isfile(yaml_path):
        return {}
    try:
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def infer_ruler_max_new_tokens(jsonl_path: str, override: int | None = None) -> int:
    if override is not None and override > 0:
        return int(override)
    task_key = os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))
    cfg = _load_ruler_synthetic_yaml()
    entry = cfg.get(task_key) or {}
    base = entry.get("task")
    if base and base in RULER_BASE_TASK_MAX_NEW_TOKENS:
        return int(RULER_BASE_TASK_MAX_NEW_TOKENS[base])
    return 128


def _postprocess_pred(text: str, dataset_name: str | None = None) -> str:
    original = text or ""
    text = original.strip()
    if "</redacted_thinking>" in text:
        text = text.split("</redacted_thinking>")[-1]
    text = re.sub(r"<redacted_thinking>.*?</redacted_thinking>", "", text, flags=re.DOTALL)
    text = text.replace("<redacted_thinking>", "").replace("</redacted_thinking>", "")
    text = text.strip()
    for prefix in ("Answer:", "答案：", "回答："):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    multiline_ok = {"gov_report", "qmsum", "multi_news", "vcsum", "lcc", "repobench-p"}
    if dataset_name not in multiline_ok:
        separators = [
            "\nAnswer:",
            "\n答案：",
            "\n回答：",
            "\n\n",
            "\nThe answer",
            "\nExplanation",
            "\nReason",
        ]
        cut_idx = None
        for sep in separators:
            idx = text.find(sep)
            if idx != -1:
                cut_idx = idx if cut_idx is None else min(cut_idx, idx)
        if cut_idx is not None:
            text = text[:cut_idx].strip()
    force_single_line = {
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "multifieldqa_zh",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "trec",
        "triviaqa",
        "samsum",
        "lsht",
        "passage_count",
        "passage_retrieval_en",
        "passage_retrieval_zh",
    }
    if dataset_name in force_single_line and "\n" in text:
        text = text.splitlines()[0].strip()
    if not text:
        fallback = re.sub(r"<redacted_thinking>.*", "", original, flags=re.DOTALL).strip()
        text = fallback.splitlines()[0].strip() if fallback else original.strip()
    return text


def _ruler_checkpoint_path(output_path: str) -> str:
    base, _ext = os.path.splitext(output_path)
    return f"{base}_checkpoint.json"


def _restore_ruler_progress(
    output_path: str,
    checkpoint_path: str,
    jsonl_path: str,
    max_samples: int,
) -> tuple[int, list[float], list[float]]:
    jsonl_abs = os.path.abspath(jsonl_path)
    prefill_ms_list: list[float] = []
    total_generate_ms_list: list[float] = []

    ckpt_mismatch = False
    if os.path.isfile(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
        except (json.JSONDecodeError, OSError):
            ckpt = None
        else:
            if ckpt and (
                ckpt.get("jsonl_path") != jsonl_abs
                or int(ckpt.get("max_samples", -1)) != int(max_samples)
            ):
                ckpt_mismatch = True
            if (
                ckpt
                and ckpt.get("jsonl_path") == jsonl_abs
                and int(ckpt.get("max_samples", -1)) == int(max_samples)
            ):
                prefill_ms_list = [float(x) for x in ckpt.get("prefill_ms", [])]
                total_generate_ms_list = [float(x) for x in ckpt.get("total_generate_ms", [])]
                return int(ckpt.get("next_index", 0)), prefill_ms_list, total_generate_ms_list

    # 如果 checkpoint 存在但不匹配当前 jsonl/max_samples，就不要从 output 推断进度，避免跨任务混跑。
    if ckpt_mismatch:
        return 0, [], []

    # 兼容：当 checkpoint 不存在时，仅在 output 每行包含匹配的 jsonl_path 字段时才允许恢复。
    if os.path.isfile(output_path):
        matched = True
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_jsonl = row.get("jsonl_path")
                if row_jsonl is None:
                    matched = False
                    break
                if os.path.abspath(str(row_jsonl)) != jsonl_abs:
                    matched = False
                    break
                if "prefill_ms" in row:
                    prefill_ms_list.append(float(row["prefill_ms"]))
                if "total_generate_ms" in row:
                    total_generate_ms_list.append(float(row["total_generate_ms"]))
        if matched and prefill_ms_list:
            return len(prefill_ms_list), prefill_ms_list, total_generate_ms_list

    return 0, [], []


def _write_ruler_checkpoint(
    checkpoint_path: str,
    jsonl_path: str,
    max_samples: int,
    next_index: int,
    max_new_tokens_cfg: int,
    max_seq_len: int,
    prefill_ms_list: list[float],
    total_generate_ms_list: list[float],
) -> None:
    payload = {
        "version": 1,
        "jsonl_path": os.path.abspath(jsonl_path),
        "max_samples": int(max_samples),
        "next_index": int(next_index),
        "max_new_tokens": int(max_new_tokens_cfg),
        "max_seq_len": int(max_seq_len),
        "prefill_ms": prefill_ms_list,
        "total_generate_ms": total_generate_ms_list,
    }
    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, checkpoint_path)


def generate_with_prefill_timing(
    model: Qwen3ForCausalLM,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float | None, float]:
    """返回 (output_ids, prefill_ms, total_generate_ms)。"""
    orig_forward = model.forward
    state: dict = {"done": False, "prefill_ms": None}

    def timed_forward(*args, **kwargs):
        if not state["done"]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = orig_forward(*args, **kwargs)
        if not state["done"]:
            torch.cuda.synchronize()
            state["prefill_ms"] = (time.perf_counter() - t0) * 1000
            state["done"] = True
        return out

    model.forward = timed_forward
    try:
        torch.cuda.synchronize()
        t_all0 = time.perf_counter()
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t_all0) * 1000
    finally:
        model.forward = orig_forward

    return output_ids, state["prefill_ms"], total_ms


def run_ruler_baseline(
    *,
    local_path: str,
    device: torch.device,
    max_seq_len: int,
    jsonl_path: str,
    output_path: str,
    max_samples: int,
    max_new_tokens_override: int | None,
    resume: bool,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not os.path.isfile(jsonl_path):
        task_hint = os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))
        raise FileNotFoundError(
            f"RULER jsonl 不存在: {jsonl_path}\n"
            f"请先用 RULER scripts/data/prepare.py 生成（任务目录名示例: {task_hint or 'niah_single_1'}）。"
        )

    max_gen = infer_ruler_max_new_tokens(jsonl_path, max_new_tokens_override)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = _ruler_checkpoint_path(output_path)
    start, prefill_ms_list, total_generate_ms_list = (0, [], [])
    if resume:
        start, prefill_ms_list, total_generate_ms_list = _restore_ruler_progress(
            output_path, checkpoint_path, jsonl_path, max_samples
        )
    else:
        if os.path.isfile(output_path):
            os.remove(output_path)
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = Qwen3ForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    run_ruler_baseline_loaded(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_seq_len=max_seq_len,
        jsonl_path=jsonl_path,
        output_path=output_path,
        max_samples=max_samples,
        max_new_tokens_override=max_new_tokens_override,
        resume=resume,
    )


def run_ruler_baseline_loaded(
    *,
    model: Qwen3ForCausalLM,
    tokenizer,
    device: torch.device,
    max_seq_len: int,
    jsonl_path: str,
    output_path: str,
    max_samples: int,
    max_new_tokens_override: int | None,
    resume: bool,
) -> None:
    """与 run_ruler_baseline 相同逻辑，但复用已加载的 tokenizer/model（适合批量 task）。"""
    if not os.path.isfile(jsonl_path):
        task_hint = os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))
        raise FileNotFoundError(
            f"RULER jsonl 不存在: {jsonl_path}\n"
            f"请先用 RULER scripts/data/prepare.py 生成（任务目录名示例: {task_hint or 'niah_single_1'}）。"
        )

    max_gen = infer_ruler_max_new_tokens(jsonl_path, max_new_tokens_override)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = _ruler_checkpoint_path(output_path)
    start, prefill_ms_list, total_generate_ms_list = (0, [], [])
    if resume:
        start, prefill_ms_list, total_generate_ms_list = _restore_ruler_progress(
            output_path, checkpoint_path, jsonl_path, max_samples
        )
        start=0
    else:
        if os.path.isfile(output_path):
            os.remove(output_path)
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)

    data = load_dataset("json", data_files=jsonl_path, split="train")
    n_total = len(data)
    end = n_total if max_samples <= 0 else min(n_total, max_samples)

    if start >= end:
        print(
            f"[RULER baseline] 无需继续：start={start}, end={end}, n_total={n_total}, max_samples={max_samples}"
        )
        return

    if start > 0 and not os.path.isfile(output_path):
        raise FileNotFoundError(
            f"断点续跑需要已有输出文件，但未找到: {output_path}（start={start}）"
        )

    mode = "a" if start > 0 else "w"
    file_obj = open(output_path, mode, encoding="utf-8")
    try:
        for i in tqdm(
            range(start, end),
            desc=os.path.basename(jsonl_path),
            total=end - start,
        ):
            sample = data[i]
            model_prompt = sample["input"]
            ref_outputs = sample.get("outputs")
            ruler_index = sample.get("index", i)

            max_input_len = max(1, int(max_seq_len) - max_gen)
            inputs = tokenizer(
                [model_prompt],
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            input_len = int(input_ids.shape[1])
            max_new_tokens_row = min(max_gen, int(max_seq_len) - input_len)

            with torch.no_grad():
                output_ids, prefill_ms, total_ms = generate_with_prefill_timing(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens_row,
                )

            prefill_ms_list.append(float(prefill_ms) if prefill_ms is not None else 0.0)
            total_generate_ms_list.append(total_ms)

            pred = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            pred = _postprocess_pred(pred, dataset_name=None)

            row = {
                "jsonl_path": os.path.abspath(jsonl_path),
                "ruler_row_index": i,
                "ruler_index": ruler_index,
                "pred": pred,
                "outputs": ref_outputs,
                "prefill_ms": prefill_ms,
                "total_generate_ms": total_ms,
                "input_len": input_len,
                "max_new_tokens": max_new_tokens_row,
                "max_new_tokens_config": max_gen,
            }
            json.dump(row, file_obj, ensure_ascii=False)
            file_obj.write("\n")
            file_obj.flush()

            _write_ruler_checkpoint(
                checkpoint_path,
                jsonl_path,
                max_samples,
                i + 1,
                max_gen,
                max_seq_len,
                prefill_ms_list,
                total_generate_ms_list,
            )

            del input_ids, attention_mask, output_ids
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    finally:
        file_obj.close()

    task_name = os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))
    n = len(prefill_ms_list)
    if n:
        print(
            f"[RULER baseline:{task_name}] prefill 平均: {sum(prefill_ms_list) / n:.1f} ms (samples={n})"
        )
    if total_generate_ms_list:
        print(
            f"[RULER baseline:{task_name}] generate 总耗时平均: "
            f"{sum(total_generate_ms_list) / len(total_generate_ms_list):.1f} ms "
            f"(samples={len(total_generate_ms_list)})"
        )

    summary_path = f"{os.path.splitext(output_path)[0]}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "Qwen3ForCausalLM",
                "benchmark": "RULER",
                "ruler_jsonl": os.path.abspath(jsonl_path),
                "task_dir_name": task_name,
                "max_samples": min(max_samples, n_total) if max_samples > 0 else n_total,
                "max_seq_len": max_seq_len,
                "max_new_tokens": max_gen,
                "tokenizer_truncation_max_len": max_seq_len - max_gen,
                "batch_size": 1,
                "dtype": "bfloat16",
                "ruler_root": os.path.abspath(RULER_ROOT),
                "checkpoint_path": os.path.abspath(checkpoint_path),
                "prefill_ms": prefill_ms_list,
                "total_generate_ms": total_generate_ms_list,
                "avg_prefill_ms": sum(prefill_ms_list) / n if n else None,
                "avg_total_generate_ms": sum(total_generate_ms_list) / len(total_generate_ms_list)
                if total_generate_ms_list
                else None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[summary] 已写入 {summary_path}")


def _split_tasks_csv(s: str) -> list[str]:
    parts = [p.strip() for p in (s or "").split(",")]
    return [p for p in parts if p]


def _list_tasks_under_save_dir(save_dir: str, subset_file: str) -> list[str]:
    tasks: list[str] = []
    try:
        for name in sorted(os.listdir(save_dir)):
            task_dir = os.path.join(save_dir, name)
            if not os.path.isdir(task_dir):
                continue
            if os.path.isfile(os.path.join(task_dir, subset_file)):
                tasks.append(name)
    except FileNotFoundError:
        return []
    return tasks


def main() -> None:
    global RULER_ROOT
    p = argparse.ArgumentParser(description="Baseline Qwen3 on RULER jsonl with prefill timing（与 example.py run_ruler_jsonl 对齐）")
    p.add_argument("--local-path", default=os.environ.get("QWEN3_LOCAL_PATH", DEFAULT_LOCAL_PATH))
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--ruler-jsonl", default=DEFAULT_RULER_JSONL)
    p.add_argument("--ruler-root", default=RULER_ROOT, help="用于解析 synthetic.yaml；也可用环境变量 RULER_ROOT")
    p.add_argument(
        "--ruler-save-dir",
        default=DEFAULT_RULER_SAVE_DIR,
        help=(
            "批量跑 task 时的数据根目录（每个 task 在其子目录下）。"
            "默认 /data/zn/ruler/4096（可用环境变量 RULER_SAVE_DIR 覆盖）。"
        ),
    )
    p.add_argument(
        "--ruler-subset-file",
        default="validation.jsonl",
        help="批量跑 task 时的子集文件名（通常 validation.jsonl / test.jsonl）。",
    )
    p.add_argument(
        "--ruler-task",
        action="append",
        default=[],
        help="要跑的 RULER task 目录名（可重复传参）。",
    )
    p.add_argument(
        "--ruler-tasks",
        default="",
        help="要跑的 task（逗号分隔），例如 niah_single_1,qa_1；等价于重复使用 --ruler-task。",
    )
    p.add_argument(
        "--ruler-all",
        action="store_true",
        help="自动扫描 --ruler-save-dir 下所有含 --ruler-subset-file 的 task 目录并逐个运行。",
    )
    p.add_argument("--output", default=DEFAULT_OUT)
    p.add_argument(
        "--ruler-output-dir",
        default="",
        help=(
            "批量跑 RULER 时每个 task 的输出目录（文件名自动为 ruler_<task>.jsonl）。"
            "为空则沿用 --output（适合单 task）。"
        ),
    )
    p.add_argument("--max-samples", type=int, default=500, help="0 表示 jsonl 全量")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=-1,
        help="覆盖 max_gen；-1 表示按任务目录名从 RULER synthetic.yaml 推断",
    )
    p.add_argument("--no-resume", action="store_true", help="清空输出与 checkpoint 重跑")
    p.add_argument("--device", default="cuda:0", help="在设置 CUDA_VISIBLE_DEVICES 后的设备名")
    args = p.parse_args()

    RULER_ROOT = os.path.abspath(args.ruler_root)

    max_new_tokens_arg = None if args.max_new_tokens < 0 else int(args.max_new_tokens)

    tasks_from_repeat = list(args.ruler_task or [])
    tasks_from_csv = _split_tasks_csv(args.ruler_tasks)
    tasks: list[str] = []
    for t in tasks_from_repeat + tasks_from_csv:
        if t not in tasks:
            tasks.append(t)

    if args.ruler_all:
        tasks = _list_tasks_under_save_dir(args.ruler_save_dir, args.ruler_subset_file)
        if not tasks:
            raise FileNotFoundError(
                f"未在 --ruler-save-dir 发现任何 task（要求存在 {args.ruler_subset_file}）：{args.ruler_save_dir}"
            )

    # 未指定批量 task：沿用原来的单 jsonl 用法。
    if not tasks:
        output_path = args.output
        # 如果用户只改了 --ruler-jsonl 但没显式改 --output，则自动按 task 名生成输出文件名，避免写到 niah 默认文件。
        if os.path.abspath(output_path) == os.path.abspath(DEFAULT_OUT):
            output_path = _auto_output_path_for_jsonl(args.ruler_jsonl, output_default=DEFAULT_OUT)
        run_ruler_baseline(
            local_path=args.local_path,
            device=torch.device(args.device),
            max_seq_len=int(args.max_seq_len),
            jsonl_path=args.ruler_jsonl,
            output_path=output_path,
            max_samples=int(args.max_samples),
            max_new_tokens_override=max_new_tokens_arg,
            resume=not args.no_resume,
        )
        return

    out_dir = (args.ruler_output_dir or "").strip()
    if not out_dir:
        raise ValueError(
            "批量跑 RULER 时请提供 --ruler-output-dir（避免多个 task 写到同一个 --output 文件）。"
        )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.local_path)
    model = Qwen3ForCausalLM.from_pretrained(
        args.local_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    for idx, task in enumerate(tasks, start=1):
        jsonl_path = os.path.join(args.ruler_save_dir, task, args.ruler_subset_file)
        output_path = os.path.join(out_dir, f"ruler_{task}.jsonl")
        print(f"\n=== [{idx}/{len(tasks)}] RULER task: {task} ===")
        run_ruler_baseline_loaded(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=int(args.max_seq_len),
            jsonl_path=jsonl_path,
            output_path=output_path,
            max_samples=int(args.max_samples),
            max_new_tokens_override=max_new_tokens_arg,
            resume=not args.no_resume,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
