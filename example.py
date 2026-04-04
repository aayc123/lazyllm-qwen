import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import Qwen3ForCausalLM, AutoTokenizer
from models import *
import torch
import json
import time
import re
from datasets import load_dataset
from tqdm import tqdm
from caches import HFCache

# RULER 仓库根目录（prepare.py 生成的 jsonl 通常在 <save_dir>/<task>/validation.jsonl）
RULER_ROOT = os.environ.get("RULER_ROOT", "/data/zn/RULER")

# 默认使用预生成的数据集目录（任务目录直接位于其下，例如 <save_dir>/qa_1/validation.jsonl）。
DEFAULT_RULER_SAVE_DIR = os.environ.get("RULER_SAVE_DIR", "/data/zn/ruler/4096")

# 与 RULER scripts/data/synthetic/constants.py 中 TASKS[*]['tokens_to_generate'] 对齐（按 task 类型）
RULER_BASE_TASK_MAX_NEW_TOKENS = {
    "niah": 128,
    "variable_tracking": 30,
    "common_words_extraction": 120,
    "freq_words_extraction": 50,
    "qa": 32,
}


def _load_ruler_synthetic_yaml():
    """加载 RULER scripts/synthetic.yaml，用于由任务名推断底层 task 类型。"""
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
    """根据 jsonl 所在目录名（如 niah_single_1）推断 max_new_tokens，与 prepare 时 tokens_to_generate 一致。"""
    if override is not None and override > 0:
        return int(override)
    task_key = os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))
    cfg = _load_ruler_synthetic_yaml()
    entry = cfg.get(task_key) or {}
    base = entry.get("task")
    if base and base in RULER_BASE_TASK_MAX_NEW_TOKENS:
        return int(RULER_BASE_TASK_MAX_NEW_TOKENS[base])
    return 128

local_path = "/data/zn/model/models/Qwen3-8B"
device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(local_path)

qwen3_model = Qwen3ForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

config = qwen3_model.config
state_dict = qwen3_model.state_dict()  # 仍在CPU

del qwen3_model
import gc; gc.collect()
print(f"del原模型后 GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

lazy_qwen3_model = LazyQwen3ForCausalLM.from_qwen3_state_dict(
    state_dict,
    config,
    pruning_rates = {
        i: 0.5 if i ==15 or i==20 or i==25 else 0.0
        for i in range(36)
    }
    # pruning_rates = {
    #     i: 0.3 if i % 10 ==0 and i>0 else 0.0
    #     for i in range(36)
    # }
    
    # pruning_rates={i: 0.03 for i in range(36)},
)
lazy_qwen3_model.eval()
del state_dict
gc.collect()
# print(f"转换后 GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# 最后才上GPU
lazy_qwen3_model = lazy_qwen3_model.to(device)
torch.cuda.empty_cache()
# print(f"上GPU后: {torch.cuda.memory_allocated()/1024**3:.1f}GB")  # 应该约15GB

from caches import KVCache, AuxCache
# 预分配 KV/Aux cache 的“总长度上限”(prompt + generation)。
# 注意：run_dataset 里会根据每个数据集的 max_gen 为 prompt 留出预算，保证不会越界。
MAX_SEQ_LEN = 35200
global_kv_cache = KVCache(
    lazy_qwen3_model.config.num_hidden_layers,
    1,  # batch_size=1
    lazy_qwen3_model.config.num_key_value_heads,
    MAX_SEQ_LEN,
    lazy_qwen3_model.config.head_dim,
    device,
    dtype=torch.bfloat16,
)
global_aux_cache = AuxCache(
    lazy_qwen3_model.config.num_hidden_layers,
    1,
    MAX_SEQ_LEN,
    lazy_qwen3_model.config.hidden_size,
    device,
    dtype=torch.bfloat16,
)
# print(f"预分配cache后: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
dataset2prompt = json.load(open("/data/zn/longbench/config/dataset2prompt.json"))
dataset2maxlen = json.load(open("/data/zn/longbench/config/dataset2maxlen.json"))


def _postprocess_pred(text: str, dataset_name: str | None = None) -> str:
    """Best-effort extraction of the final answer.

    Qwen-family chat models may emit <think>...</think> blocks. LongBench expects
    only the final answer string.
    """
    original = text or ""
    text = original.strip()

    # If there's an explicit closing tag, keep the content after the last one.
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Remove any remaining think blocks/tags.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip()

    # Strip common prefixes.
    for prefix in ("Answer:", "答案：", "回答："):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # For most LongBench tasks, the expected output is just the answer (no self-eval
    # like "The answer is correct"), but some models may keep generating and repeat
    # "Answer:" blocks. For non-multiline tasks, truncate at common separators.
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

    # Only force single-line for datasets that are evaluated as single-line in LongBench.
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

    # Fallback if we stripped everything.
    if not text:
        fallback = re.sub(r"<think>.*", "", original, flags=re.DOTALL).strip()
        text = fallback.splitlines()[0].strip() if fallback else original.strip()

    return text

def run_dataset(dataset_name, output_path, max_samples=50, kv_cache=None, aux_cache=None):
    data_file = f"/data/zn/longbench/data/{dataset_name}.jsonl"
    data = load_dataset('json', data_files=data_file, split='train')
    prompt_format = dataset2prompt[dataset_name]
    max_gen = int(dataset2maxlen[dataset_name])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prefill_ms_list = []
    total_generate_ms_list = []

    with open(output_path, "w") as f:
        for i, sample in enumerate(tqdm(data, desc=dataset_name)):
            if max_samples > 0 and i >= max_samples:
                break

            prompt = prompt_format.format(**sample)
            # LongBench prompts already contain task instructions and an explicit
            # "Answer:" suffix. Feeding them directly avoids chat-style preambles.
            model_prompt = prompt

            # 关键：给 generation 预留 token 预算，避免 input_len + max_gen > MAX_SEQ_LEN 导致 cache 越界。
            max_input_len = max(1, int(MAX_SEQ_LEN) - max_gen)
            inputs = tokenizer(
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            input_len = int(input_ids.shape[1])
            max_new_tokens_row = min(200, int(MAX_SEQ_LEN) - input_len)

            torch.cuda.reset_peak_memory_stats()
            output_ids, timings = lazy_qwen3_model.generate(
                input_ids,
                attention_mask=attention_mask,
                # 总长度上限直接用预分配上限，保证不会超过 cache 容量
                max_length=int(MAX_SEQ_LEN),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                preallocated_kv_cache=kv_cache,
                preallocated_aux_cache=aux_cache,
                return_timings=True,
            )

            prefill_ms_list.append(timings["prefill_ms"])
            total_generate_ms_list.append(timings["total_generate_ms"])

            pred = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            pred = _postprocess_pred(pred, dataset_name=dataset_name)

            del input_ids, attention_mask, output_ids
            import gc; gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            json.dump(
                {
                    "pred": pred,
                    "answers": sample["answers"],
                    "all_classes": sample.get("all_classes"),
                    "length": sample["length"],
                    "prefill_ms": timings["prefill_ms"],
                    "total_generate_ms": timings["total_generate_ms"],
                    "input_len": input_len,
                    "max_new_tokens": max_new_tokens_row,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    n = len(prefill_ms_list)
    avg_p = sum(prefill_ms_list) / n if n else 0.0
    avg_t = sum(total_generate_ms_list) / len(total_generate_ms_list) if total_generate_ms_list else 0.0
    print(
        f"[{dataset_name}] prefill 平均: {avg_p:.1f} ms, generate 总耗时平均: {avg_t:.1f} ms, samples: {n}"
    )

    base, _ext = os.path.splitext(output_path)
    summary_path = f"{base}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": "LazyQwen3ForCausalLM",
                "dataset": dataset_name,
                "max_samples": n,
                "max_seq_len": int(MAX_SEQ_LEN),
                "max_gen_dataset_config": max_gen,
                "tokenizer_truncation_max_len": int(MAX_SEQ_LEN) - max_gen,
                "lazy_compat_max_new_tokens": "min(200, MAX_SEQ_LEN - input_len)",
                "batch_size": 1,
                "dtype": "bfloat16",
                "prefill_ms": prefill_ms_list,
                "total_generate_ms": total_generate_ms_list,
                "avg_prefill_ms": avg_p,
                "avg_total_generate_ms": avg_t,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[summary] 已写入 {summary_path}")


def _ruler_checkpoint_path(output_path: str) -> str:
    base, _ext = os.path.splitext(output_path)
    return f"{base}_checkpoint.json"


def _restore_ruler_progress(
    output_path: str,
    checkpoint_path: str,
    jsonl_path: str,
    max_samples: int,
) -> tuple[int, list[float], list[float]]:
    """断点续跑：优先读 checkpoint；否则从已有 jsonl 行恢复。"""
    jsonl_abs = os.path.abspath(jsonl_path)
    prefill_ms_list: list[float] = []
    total_generate_ms_list: list[float] = []

    if os.path.isfile(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
        except (json.JSONDecodeError, OSError):
            ckpt = None
        else:
            if (
                ckpt
                and ckpt.get("jsonl_path") == jsonl_abs
                and int(ckpt.get("max_samples", -1)) == int(max_samples)
            ):
                prefill_ms_list = [float(x) for x in ckpt.get("prefill_ms", [])]
                total_generate_ms_list = [float(x) for x in ckpt.get("total_generate_ms", [])]
                return int(ckpt.get("next_index", 0)), prefill_ms_list, total_generate_ms_list

    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "prefill_ms" in row:
                    prefill_ms_list.append(float(row["prefill_ms"]))
                if "total_generate_ms" in row:
                    total_generate_ms_list.append(float(row["total_generate_ms"]))
        return len(prefill_ms_list), prefill_ms_list, total_generate_ms_list

    return 0, [], []


def _write_ruler_checkpoint(
    checkpoint_path: str,
    jsonl_path: str,
    max_samples: int,
    next_index: int,
    max_new_tokens: int,
    prefill_ms_list: list[float],
    total_generate_ms_list: list[float],
) -> None:
    payload = {
        "version": 1,
        "jsonl_path": os.path.abspath(jsonl_path),
        "max_samples": int(max_samples),
        "next_index": int(next_index),
        "max_new_tokens": int(max_new_tokens),
        "max_seq_len": int(MAX_SEQ_LEN),
        "prefill_ms": prefill_ms_list,
        "total_generate_ms": total_generate_ms_list,
    }
    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, checkpoint_path)


def run_ruler_jsonl(
    jsonl_path: str,
    output_path: str,
    max_samples: int = 500,
    max_new_tokens: int | None = None,
    kv_cache=None,
    aux_cache=None,
    resume: bool = True,
) -> None:
    """使用 RULER 格式 jsonl（字段 index / input / outputs）跑 LazyQwen3，记录 prefill，并支持 checkpoint 续跑。

    数据需先用 RULER 的 scripts/data/prepare.py 生成，例如：
    ``python prepare.py --save_dir <dir> --task niah_single_1 --tokenizer_path ... --max_seq_length ...``
    对应样本在 ``<save_dir>/niah_single_1/validation.jsonl``。
    """
    if not os.path.isfile(jsonl_path):
        task_hint = os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))
        raise FileNotFoundError(
            f"RULER jsonl 不存在: {jsonl_path}\n\n"
            "请先在已安装 transformers 的环境里生成数据（与跑 Lazy 模型同一 conda/env 最省事），例如：\n"
            "  pip install wonderwords pyyaml tenacity  # NIAH 等任务需要 wonderwords\n"
            "  cd /data/zn/RULER/scripts/data\n"
            "  python3 prepare.py \\\n"
            "    --save_dir /data/zn/RULER/data \\\n"
            "    --benchmark synthetic \\\n"
            f"    --task {task_hint or 'niah_single_1'} \\\n"
            "    --tokenizer_path /data/zn/model/models/Qwen3-8B \\\n"
            "    --tokenizer_type hf \\\n"
            "    --max_seq_length 4096 \\\n"
            "    --num_samples 30 \\\n"
            "    --subset validation \\\n"
            "    --model_template_type base\n\n"
            "生成成功后应出现: <save_dir>/<task>/validation.jsonl\n"
            "也可用: python Lazy-Llama/example.py --ruler-jsonl <你的.jsonl路径>"
        )

    max_gen = infer_ruler_max_new_tokens(jsonl_path, max_new_tokens)
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
        start = 0
        prefill_ms_list = []
        total_generate_ms_list = []
        if os.path.isfile(output_path):
            os.remove(output_path)
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)

    data = load_dataset("json", data_files=jsonl_path, split="train")
    n_total = len(data)
    end = n_total if max_samples <= 0 else min(n_total, max_samples)

    if start >= end:
        print(
            f"[RULER] 无需继续：start={start}, 目标行上界 end={end}, 数据行数={n_total}, max_samples={max_samples}"
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

            max_input_len = max(1, int(MAX_SEQ_LEN) - max_gen)
            inputs = tokenizer(
                [model_prompt],
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            input_len = int(input_ids.shape[1])
            max_new_tokens_row = min(max_gen, int(MAX_SEQ_LEN) - input_len)

            torch.cuda.reset_peak_memory_stats()
            output_ids, timings = lazy_qwen3_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=int(MAX_SEQ_LEN),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                preallocated_kv_cache=kv_cache,
                preallocated_aux_cache=aux_cache,
                return_timings=True,
            )

            prefill_ms_list.append(timings["prefill_ms"])
            total_generate_ms_list.append(timings["total_generate_ms"])

            pred = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            pred = _postprocess_pred(pred, dataset_name=None)

            del input_ids, attention_mask, output_ids
            import gc

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            out_row = {
                "ruler_row_index": i,
                "ruler_index": ruler_index,
                "pred": pred,
                "outputs": ref_outputs,
                "prefill_ms": timings["prefill_ms"],
                "total_generate_ms": timings["total_generate_ms"],
                "input_len": input_len,
                "max_new_tokens": max_new_tokens_row,
                "max_new_tokens_config": max_gen,
            }
            json.dump(out_row, file_obj, ensure_ascii=False)
            file_obj.write("\n")
            file_obj.flush()

            next_index = i + 1
            _write_ruler_checkpoint(
                checkpoint_path,
                jsonl_path,
                max_samples,
                next_index,
                max_gen,
                prefill_ms_list,
                total_generate_ms_list,
            )
    finally:
        file_obj.close()

    n = len(prefill_ms_list)
    avg_p = sum(prefill_ms_list) / n if n else 0.0
    avg_t = sum(total_generate_ms_list) / len(total_generate_ms_list) if total_generate_ms_list else 0.0
    task_name = os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))
    print(
        f"[RULER:{task_name}] prefill 平均: {avg_p:.1f} ms, generate 总耗时平均: {avg_t:.1f} ms, 已完成样本数: {n}"
    )

    summary_path = f"{os.path.splitext(output_path)[0]}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "LazyQwen3ForCausalLM",
                "benchmark": "RULER",
                "ruler_jsonl": os.path.abspath(jsonl_path),
                "task_dir_name": task_name,
                "max_samples": min(max_samples, n_total) if max_samples > 0 else n_total,
                "max_seq_len": int(MAX_SEQ_LEN),
                "max_new_tokens": max_gen,
                "tokenizer_truncation_max_len": int(MAX_SEQ_LEN) - max_gen,
                "batch_size": 1,
                "dtype": "bfloat16",
                "ruler_root": os.path.abspath(RULER_ROOT),
                "checkpoint_path": os.path.abspath(checkpoint_path),
                "prefill_ms": prefill_ms_list,
                "total_generate_ms": total_generate_ms_list,
                "avg_prefill_ms": avg_p,
                "avg_total_generate_ms": avg_t,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[summary] 已写入 {summary_path}")


# print("=== 基准测试：原始Qwen3 ===")
# baseline_model = Qwen3ForCausalLM.from_pretrained(
#     local_path, torch_dtype=torch.float32, device_map=device
# )
# baseline_model.eval()

# from datasets import load_dataset
# import json
# data = load_dataset('json', data_files="/data/zn/longbench/data/qasper.jsonl", split='train')
# dataset2prompt = json.load(open("/data/zn/longbench/config/dataset2prompt.json"))
# sample = data[0]
# prompt = dataset2prompt["qasper"].format(**sample)
# messages = [{"role": "user", "content": prompt}]
# model_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = tokenizer([model_prompt], return_tensors="pt", truncation=True, max_length=35000).to(device)

# with torch.no_grad():
#     out = baseline_model.generate(
#         **inputs,
#         max_new_tokens=50,
#         do_sample=False,
#     )
# pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# print(f"[baseline] pred: {pred}")

# del baseline_model
# gc.collect()
# torch.cuda.empty_cache()
# print("=== 基准测试结束 ===")
def _run_baseline_compare_short() -> None:
    """可选：加载一次 baseline Qwen3 与 Lazy 做短序列对比（较慢，默认不跑）。"""
    print("=== 对比测试 ===")
    test_text = (
        "Please summarize the following paper abstract in one sentence: Recent advances in large language models "
        "have shown remarkable capabilities across diverse tasks. However, these models require substantial "
        "computational resources during inference, particularly for long contexts. We propose a novel dynamic "
        "token pruning method that selectively processes only the most relevant tokens at each layer."
    )
    inputs = tokenizer([test_text], return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print(f"test input_ids: {input_ids}, shape: {input_ids.shape}")

    baseline = Qwen3ForCausalLM.from_pretrained(local_path, torch_dtype=torch.float32).to(device)
    baseline.eval()
    with torch.no_grad():

        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:
                print(
                    f"[baseline lm_head input] shape={hs.shape}, last token: mean={hs[0,-1,:].mean().item():.4f}, "
                    f"std={hs[0,-1,:].std().item():.4f}, max={hs[0,-1,:].max().item():.4f}"
                )

        hook = baseline.model.norm.register_forward_hook(hook_fn)
        baseline_out = baseline(input_ids=input_ids, attention_mask=attention_mask)
        baseline_logits = baseline_out.logits[0, -1, :]
        baseline_top3 = torch.topk(baseline_logits, 3)
        hook.remove()
        layer_outputs_baseline = {}
        hooks = []
        for i, layer in enumerate(baseline.model.layers):

            def make_hook(idx):
                def hook_layer(m, inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                    else:
                        hs = out
                    layer_outputs_baseline[idx] = hs[0, -1, :5].tolist()

                return hook_layer

            hooks.append(layer.register_forward_hook(make_hook(i)))

        baseline(input_ids=input_ids, attention_mask=attention_mask)
        for h in hooks:
            h.remove()

        for i in range(36):
            print(f"[baseline layer{i} last token] {layer_outputs_baseline[i]}")
        print(
            f"[baseline] top3 ids={baseline_top3.indices.tolist()}, "
            f"vals={[f'{v:.3f}' for v in baseline_top3.values.tolist()]}"
        )
        print(f"[baseline] top1 token: '{tokenizer.decode([baseline_top3.indices[0].item()])}'")
        print(
            f"[baseline lm_head] weight sum={baseline.lm_head.weight.float().sum().item():.4f}, "
            f"std={baseline.lm_head.weight.float().std().item():.6f}"
        )
        print(f"[baseline norm] weight sum={baseline.model.norm.weight.float().sum().item():.4f}")
    del baseline
    print(f"[lazy norm] weight sum={lazy_qwen3_model.model.norm.weight.float().sum().item():.4f}")
    print(
        f"[lazy lm_head] weight sum={lazy_qwen3_model.lm_head.weight.float().sum().item():.4f}, "
        f"std={lazy_qwen3_model.lm_head.weight.float().std().item():.6f}"
    )
    torch.cuda.empty_cache()

    global_kv_cache.reset()
    global_aux_cache.reset()
    with torch.no_grad():
        lazy_qwen3_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            preallocated_kv_cache=global_kv_cache,
            preallocated_aux_cache=global_aux_cache,
        )
    print("=== 对比测试结束 ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="LazyQwen3：RULER 或 LongBench 评测，记录 prefill，支持断点续跑。")
    parser.add_argument(
        "--mode",
        choices=["ruler", "longbench"],
        default="ruler",
        help="ruler：使用 RULER 的 jsonl（input/outputs）；longbench：沿用原 LongBench 配置。",
    )
    parser.add_argument(
        "--ruler-jsonl",
        type=str,
        default=os.path.join(DEFAULT_RULER_SAVE_DIR, "niah_single_1", "validation.jsonl"),
        help=(
            "RULER 格式 jsonl，例如 <save_dir>/niah_single_1/validation.jsonl。"
            "默认 /data/zn/ruler/4096/niah_single_1/validation.jsonl（可用环境变量 RULER_SAVE_DIR 覆盖）。"
        ),
    )
    parser.add_argument(
        "--ruler-save-dir",
        type=str,
        default=DEFAULT_RULER_SAVE_DIR,
        help=(
            "RULER prepare.py 的 --save_dir（用于按 task 名拼出 jsonl 路径）。"
            "默认 /data/zn/ruler/4096（可用环境变量 RULER_SAVE_DIR 覆盖）。"
        ),
    )
    parser.add_argument(
        "--ruler-subset-file",
        type=str,
        default="validation.jsonl",
        help="RULER task 目录下的子集文件名（通常为 validation.jsonl / test.jsonl）。",
    )
    parser.add_argument(
        "--ruler-task",
        action="append",
        default=[],
        help=(
            "要跑的 RULER task 目录名（可重复传参）。例如：--ruler-task niah_single_1 --ruler-task qa_1k"
        ),
    )
    parser.add_argument(
        "--ruler-tasks",
        type=str,
        default="",
        help="要跑的 task（逗号分隔），例如 niah_single_1,qa_1k；等价于重复使用 --ruler-task。",
    )
    parser.add_argument(
        "--ruler-all",
        action="store_true",
        help="自动扫描 --ruler-save-dir 下所有含 --ruler-subset-file 的 task 目录并逐个运行。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/zn/Lazy-Llama/pred/lazy_qwen3_50/ruler_niah_single_1.jsonl",
        help="预测输出 jsonl；同路径下会写 _checkpoint.json 与 _summary.json",
    )
    parser.add_argument(
        "--ruler-output-dir",
        type=str,
        default="",
        help=(
            "批量跑 RULER 时每个 task 的输出目录（文件名自动为 ruler_<task>.jsonl）。"
            "为空则沿用 --output（适合单任务）。"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=500, help="最多跑前 N 条；0 表示全量。")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=-1,
        help="生成上限；-1 表示按 RULER 任务类型自动推断（与 prepare 时 tokens_to_generate 对齐）。",
    )
    parser.add_argument("--no-resume", action="store_true", help="忽略断点，清空输出与 checkpoint 重跑。")
    parser.add_argument(
        "--longbench-dataset",
        type=str,
        default="qasper",
        help="--mode longbench 时的数据集名（对应 longbench/data/<name>.jsonl）。",
    )
    parser.add_argument(
        "--longbench-output",
        type=str,
        default="/data/zn/Lazy-Llama/pred/lazy_qwen3_50/qasper_30.jsonl",
        help="--mode longbench 时的输出路径。",
    )
    parser.add_argument(
        "--run-baseline-compare",
        action="store_true",
        help="运行前先做短序列 baseline vs lazy 对比（会多加载一份 Qwen3，较慢）。",
    )
    args = parser.parse_args()

    if args.run_baseline_compare:
        _run_baseline_compare_short()

    global_kv_cache.reset()
    global_aux_cache.reset()

    max_new_tokens_arg = None if args.max_new_tokens < 0 else int(args.max_new_tokens)

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

    if args.mode == "ruler":
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

        # 兼容旧用法：未指定 task / all 时，按 --ruler-jsonl 单文件运行。
        if not tasks:
            run_ruler_jsonl(
                args.ruler_jsonl,
                args.output,
                max_samples=int(args.max_samples),
                max_new_tokens=max_new_tokens_arg,
                kv_cache=global_kv_cache,
                aux_cache=global_aux_cache,
                resume=not args.no_resume,
            )
            return

        out_dir = (args.ruler_output_dir or "").strip()
        if not out_dir:
            raise ValueError(
                "批量跑 RULER 时请提供 --ruler-output-dir（避免多个 task 写到同一个 --output 文件）。"
            )
        os.makedirs(out_dir, exist_ok=True)

        for idx, task in enumerate(tasks, start=1):
            jsonl_path = os.path.join(args.ruler_save_dir, task, args.ruler_subset_file)
            output_path = os.path.join(out_dir, f"ruler_{task}.jsonl")
            print(f"\n=== [{idx}/{len(tasks)}] RULER task: {task} ===")
            global_kv_cache.reset()
            global_aux_cache.reset()
            run_ruler_jsonl(
                jsonl_path,
                output_path,
                max_samples=int(args.max_samples),
                max_new_tokens=max_new_tokens_arg,
                kv_cache=global_kv_cache,
                aux_cache=global_aux_cache,
                resume=not args.no_resume,
            )
    else:
        run_dataset(
            args.longbench_dataset,
            args.longbench_output,
            max_samples=int(args.max_samples),
            kv_cache=global_kv_cache,
            aux_cache=global_aux_cache,
        )


if __name__ == "__main__":
    main()