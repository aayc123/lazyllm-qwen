import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json
import time

import torch
from datasets import load_dataset
from transformers import Qwen3ForCausalLM, AutoTokenizer

from models import LazyQwen3ForCausalLM
from caches import KVCache, AuxCache


# 可按需修改
LOCAL_PATH = "/data/zn/model/models/Qwen3-8B"
DATA_FILE = "/data/zn/longbench/data/qasper.jsonl"
DATASET2PROMPT_PATH = "/data/zn/longbench/config/dataset2prompt.json"
DATASET2MAXLEN_PATH = "/data/zn/longbench/config/dataset2maxlen.json"
SAMPLE_INDEX = 0  # 对第几条样本做对比
MAX_INPUT_LEN = 5200  # 与 example.py 保持一致的截断长度


def build_prompt(sample, dataset_name="qasper"):
    with open(DATASET2PROMPT_PATH, "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    prompt_format = dataset2prompt[dataset_name]
    return prompt_format.format(**sample)


def load_sample(idx=SAMPLE_INDEX):
    data = load_dataset("json", data_files=DATA_FILE, split="train")
    sample = data[idx]
    return sample


def run_baseline_qwen3(tokenizer, device, model_prompt, max_gen):
    print("\n===== Baseline Qwen3 =====")
    baseline = Qwen3ForCausalLM.from_pretrained(
        LOCAL_PATH,
        torch_dtype=torch.float32,
    ).to(device)
    baseline.eval()

    inputs = tokenizer([model_prompt], return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(device)

    with torch.no_grad():
        # Prefill：直接 forward 一次拿到最后一个 token 的 logits
        prefill_out = baseline(**inputs, use_cache=True)
        baseline_prefill_logits = prefill_out.logits[0, -1, :].float().cpu()
        del prefill_out

        torch.cuda.synchronize(device)
        t0 = time.time()
        output_ids = baseline.generate(
            **inputs,
            max_new_tokens=max_gen,
            do_sample=False,
        )
        torch.cuda.synchronize(device)
        t1 = time.time()

    pred = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[Baseline] time: {(t1 - t0) * 1000:.1f} ms, new_tokens: {output_ids.shape[1] - inputs['input_ids'].shape[1]}")
    print("[Baseline] pred:\n" + pred)

    # 释放 baseline 显存，避免影响 Lazy 对比
    del baseline, inputs, output_ids
    torch.cuda.empty_cache()

    return pred, baseline_prefill_logits


def run_lazy_qwen3(tokenizer, device, model_prompt, max_gen):
    print("\n===== LazyQwen3 =====")
    # 先在 CPU 上加载一次原始模型，只取 state_dict 和 config
    qwen3_model = Qwen3ForCausalLM.from_pretrained(
        LOCAL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    state_dict = qwen3_model.state_dict()
    config = qwen3_model.config
    del qwen3_model
    torch.cuda.empty_cache()

    lazy_model = LazyQwen3ForCausalLM.from_qwen3_state_dict(
        state_dict,
        config,
        pruning_rates={i: 0 for i in range(config.num_hidden_layers)},
    ).to(device)
    lazy_model.eval()
    # print(f"rotary_emb inv_freq device: {lazy_model.model.rotary_emb.inv_freq.device}")
    # print(f"rotary_emb inv_freq dtype: {lazy_model.model.rotary_emb.inv_freq.dtype}")
    del state_dict

    inputs = tokenizer([model_prompt], return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 根据当前样本动态设定 cache 长度
    seq_len = input_ids.shape[1]
    max_length = seq_len + max_gen

    kv_cache = KVCache(
        lazy_model.config.num_hidden_layers,
        1,
        lazy_model.config.num_key_value_heads,
        max_length,
        lazy_model.config.head_dim,
        device,
        dtype=torch.float32,
    )
    aux_cache = AuxCache(
        lazy_model.config.num_hidden_layers,
        1,
        max_length,
        lazy_model.config.hidden_size,
        device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        # Prefill：通过 Lazy 模型的 forward 拿到最后一个 token 的 logits
        cache_position = torch.arange(seq_len, device=device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        lazy_logits, _ = lazy_model(
            kv_cache=kv_cache,
            aux_cache=aux_cache,
            cache_position=cache_position,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        lazy_prefill_logits = lazy_logits[0, -1, :].float().cpu()

        torch.cuda.synchronize(device)
        t0 = time.time()
        output_ids = lazy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            preallocated_kv_cache=kv_cache,
            preallocated_aux_cache=aux_cache,
        )
        torch.cuda.synchronize(device)
        t1 = time.time()

    pred = tokenizer.decode(output_ids[0][seq_len:], skip_special_tokens=True)
    print(f"[Lazy] time: {(t1 - t0) * 1000:.1f} ms, new_tokens: {output_ids.shape[1] - seq_len}")
    print("[Lazy] pred:\n" + pred)

    # 清理显存
    del lazy_model, kv_cache, aux_cache, inputs, input_ids, attention_mask, output_ids
    torch.cuda.empty_cache()

    return pred, lazy_prefill_logits


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

    with open(DATASET2MAXLEN_PATH, "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)
    max_gen = int(dataset2maxlen["qasper"])

    sample = load_sample(SAMPLE_INDEX)
    prompt = build_prompt(sample, "qasper")
    messages = [{"role": "user", "content": prompt}]
    model_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"Sample index: {SAMPLE_INDEX}")
    print("Question:", sample.get("input"))
    print("Ground truth answers:", sample.get("answers"))

    baseline_pred, baseline_prefill_logits = run_baseline_qwen3(tokenizer, device, model_prompt, max_gen)
    lazy_pred, lazy_prefill_logits = run_lazy_qwen3(tokenizer, device, model_prompt, max_gen)

    # Prefill logits L2 对比
    assert baseline_prefill_logits.shape == lazy_prefill_logits.shape
    dist = torch.norm(baseline_prefill_logits - lazy_prefill_logits, p=2).item()
    print(f"\n>>> Prefill Logits L2 Distance (baseline vs lazy): {dist:.6f}")

    print("\n===== Summary =====")
    print("Baseline pred:\n", baseline_pred)
    print("\nLazy pred:\n", lazy_pred)


if __name__ == "__main__":
    main()
