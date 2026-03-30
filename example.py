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
    pruning_rates={i: 0.1 for i in range(36)},
)
lazy_qwen3_model.eval()
del state_dict
gc.collect()
print(f"转换后 GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# 最后才上GPU
lazy_qwen3_model = lazy_qwen3_model.to(device)
torch.cuda.empty_cache()
print(f"上GPU后: {torch.cuda.memory_allocated()/1024**3:.1f}GB")  # 应该约15GB

from caches import KVCache, AuxCache
# 预分配 KV/Aux cache 的“总长度上限”(prompt + generation)。
# 注意：run_dataset 里会根据每个数据集的 max_gen 为 prompt 留出预算，保证不会越界。
MAX_SEQ_LEN = 5200
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
    max_gen = dataset2maxlen[dataset_name]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ttft_list = []
    
    with open(output_path, "w") as f:
        for i, sample in enumerate(tqdm(data, desc=dataset_name)):
            if max_samples > 0 and i >= max_samples:
                break

            prompt = prompt_format.format(**sample)
            # LongBench prompts already contain task instructions and an explicit
            # "Answer:" suffix. Feeding them directly avoids chat-style preambles.
            model_prompt = prompt

            # 关键：给 generation 预留 token 预算，避免 input_len + max_gen > MAX_SEQ_LEN 导致 cache 越界。
            max_input_len = max(1, int(MAX_SEQ_LEN) - int(max_gen))
            inputs = tokenizer(
                [model_prompt],
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            print(f"[before generate] allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            torch.cuda.reset_peak_memory_stats()
            output_ids = lazy_qwen3_model.generate(
                input_ids,
                attention_mask=attention_mask,
                # 总长度上限直接用预分配上限，保证不会超过 cache 容量
                max_length=int(MAX_SEQ_LEN),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                preallocated_kv_cache=kv_cache,
                preallocated_aux_cache=aux_cache,
            )
            
            pred = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            pred = _postprocess_pred(pred, dataset_name=dataset_name)
            torch.cuda.synchronize()
            t1 = time.time()
            ttft_list.append(t1 - t0)
            
            del input_ids, attention_mask, output_ids
            import gc; gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # print(f"[sample {i}] allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            # print(f"[sample {i}] reserved:  {torch.cuda.memory_reserved()/1024**3:.1f}GB")
            # print(f"[sample {i}] after cleanup GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            
            json.dump({
                "pred": pred,
                "answers": sample["answers"],
                "all_classes": sample.get("all_classes"),
                "length": sample["length"],
            }, f, ensure_ascii=False)
            f.write("\n")

    print(f"[{dataset_name}] avg time: {sum(ttft_list)/len(ttft_list)*1000:.1f}ms, samples: {len(ttft_list)}")


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
print("=== 对比测试 ===")
import torch

test_text = "Please summarize the following paper abstract in one sentence: Recent advances in large language models have shown remarkable capabilities across diverse tasks. However, these models require substantial computational resources during inference, particularly for long contexts. We propose a novel dynamic token pruning method that selectively processes only the most relevant tokens at each layer."
inputs = tokenizer([test_text], return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
print(f"test input_ids: {input_ids}, shape: {input_ids.shape}")

# 原始模型输出
from transformers import Qwen3ForCausalLM
baseline = Qwen3ForCausalLM.from_pretrained(local_path, torch_dtype=torch.float32).to(device)
baseline.eval()
with torch.no_grad():
    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] > 1:
            print(f"[baseline lm_head input] shape={hs.shape}, last token: mean={hs[0,-1,:].mean().item():.4f}, std={hs[0,-1,:].std().item():.4f}, max={hs[0,-1,:].max().item():.4f}")
    hook = baseline.model.norm.register_forward_hook(hook_fn)
    baseline_out = baseline(input_ids=input_ids, attention_mask=attention_mask)
    baseline_logits = baseline_out.logits[0, -1, :]
    baseline_top3 = torch.topk(baseline_logits, 3)
    hook.remove()
    # example.py 里，在 baseline 推理时加
    layer_outputs_baseline = {}
    hooks = []
    for i, layer in enumerate(baseline.model.layers):
        def make_hook(idx):
            def hook(m, inp, out):
                # Qwen3DecoderLayer输出是tuple，第一个元素才是hidden_states
                if isinstance(out, tuple):
                    hs = out[0]
                else:
                    hs = out
                # hs shape: (batch, seq_len, hidden_size)
                layer_outputs_baseline[idx] = hs[0, -1, :5].tolist()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    baseline_out = baseline(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()

    # 打印baseline各层最后token的前5维
    for i in range(36):
        print(f"[baseline layer{i} last token] {layer_outputs_baseline[i]}")
    print(f"[baseline] top3 ids={baseline_top3.indices.tolist()}, vals={[f'{v:.3f}' for v in baseline_top3.values.tolist()]}")
    print(f"[baseline] top1 token: '{tokenizer.decode([baseline_top3.indices[0].item()])}'")
    print(f"[baseline lm_head] weight sum={baseline.lm_head.weight.float().sum().item():.4f}, std={baseline.lm_head.weight.float().std().item():.6f}")
    print(f"[baseline norm] weight sum={baseline.model.norm.weight.float().sum().item():.4f}")
del baseline
print(f"[lazy norm] weight sum={lazy_qwen3_model.model.norm.weight.float().sum().item():.4f}")
print(f"[lazy lm_head] weight sum={lazy_qwen3_model.lm_head.weight.float().sum().item():.4f}, std={lazy_qwen3_model.lm_head.weight.float().std().item():.6f}")
torch.cuda.empty_cache()

# lazy模型输出
global_kv_cache.reset()
global_aux_cache.reset()
with torch.no_grad():
    lazy_out = lazy_qwen3_model.generate(
        input_ids, attention_mask=attention_mask,
        max_length=input_ids.shape[1]+1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        preallocated_kv_cache=global_kv_cache,
        preallocated_aux_cache=global_aux_cache,
    )
# 我们需要看prefill的logits，在generate里已经打印了
print("=== 对比测试结束 ===")
# 跑单个数据集测试
run_dataset("qasper", "/data/zn/pred/lazy_qwen3/qasper_01.jsonl",max_samples=30,kv_cache=global_kv_cache, aux_cache=global_aux_cache)