import json
import time

import torch
from datasets import load_dataset
from transformers import Qwen3ForCausalLM, AutoTokenizer

from models import LazyQwen3ForCausalLM
from caches import KVCache, AuxCache

# 跟 baseline_qasper_compare 保持一致的配置
LOCAL_PATH = "/data/zn/model/models/Qwen3-8B"
DATA_FILE = "/data/zn/longbench/data/qasper.jsonl"
DATASET2PROMPT_PATH = "/data/zn/longbench/config/dataset2prompt.json"
DATASET2MAXLEN_PATH = "/data/zn/longbench/config/dataset2maxlen.json"
SAMPLE_INDEX = 0
MAX_INPUT_LEN = 5200
MAX_DEBUG_NEW_TOKENS = 10  # 只对齐前若干个 decode token
DEBUG_MAX_INPUT_LEN = 1024  # 对齐调试时单独的截断长度，减小显存和时间开销


def build_prompt(sample, dataset_name="qasper"):
    with open(DATASET2PROMPT_PATH, "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    prompt_format = dataset2prompt[dataset_name]
    return prompt_format.format(**sample)


def load_sample(idx=SAMPLE_INDEX):
    data = load_dataset("json", data_files=DATA_FILE, split="train")
    sample = data[idx]
    return sample


def run_baseline_stepwise(tokenizer, device, model_prompt, max_new_tokens):
    print("\n===== Baseline stepwise =====")
    # 为避免与 Lazy 共用显存导致 OOM，这里直接在 CPU 上跑 Baseline
    baseline = Qwen3ForCausalLM.from_pretrained(
        LOCAL_PATH,
        torch_dtype=torch.float32,
    )
    baseline.eval()

    inputs = tokenizer([model_prompt], return_tensors="pt", truncation=True, max_length=DEBUG_MAX_INPUT_LEN)

    # Prefill 阶段：直接跑一次 forward 拿到最后一个 token 的 logits
    with torch.no_grad():
        prefill_out = baseline(**inputs, use_cache=True)
        baseline_prefill_logits = prefill_out.logits[0, -1, :].float().cpu()
        del prefill_out

        t0 = time.time()
        out = baseline.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        t1 = time.time()

    scores = [s[0].float().cpu() for s in out.scores]  # list[steps] of (vocab,)
    seq = out.sequences[0]
    input_len = inputs["input_ids"].shape[1]
    new_tokens = seq[input_len:]

    print(f"[Baseline] time: {(t1 - t0) * 1000:.1f} ms, new_tokens: {len(new_tokens)} (debug mode)")

    # 为节省显存，释放 baseline
    del baseline, out
    torch.cuda.empty_cache()

    return scores, new_tokens, inputs, baseline_prefill_logits


def run_lazy_stepwise(tokenizer, device, model_prompt, max_new_tokens):
    print("\n===== Lazy stepwise =====")
    # 先在 CPU 上加载一次原始 Qwen3，只取 state_dict 和 config
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
    del state_dict

    inputs = tokenizer([model_prompt], return_tensors="pt", truncation=True, max_length=DEBUG_MAX_INPUT_LEN)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    seq_len = input_ids.shape[1]
    max_length = seq_len + max_new_tokens

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

    # Prefill 阶段：通过 Lazy 模型的 forward 拿到最后一个 token 的 logits
    with torch.no_grad():
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
        output_ids, scores = lazy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            preallocated_kv_cache=kv_cache,
            preallocated_aux_cache=aux_cache,
            return_scores=True,
        )
        torch.cuda.synchronize(device)
        t1 = time.time()

    new_tokens = output_ids[0][seq_len:]
    scores = [s[0].float() for s in scores]  # list[steps] of (vocab,)

    print(f"[Lazy] time: {(t1 - t0) * 1000:.1f} ms, new_tokens: {len(new_tokens)} (debug mode)")

    # 清理显存
    del lazy_model, kv_cache, aux_cache
    torch.cuda.empty_cache()

    return scores, new_tokens, inputs, lazy_prefill_logits


def compare_scores(b_scores, l_scores, tokenizer, b_new_tokens, l_new_tokens):
    steps = min(len(b_scores), len(l_scores), MAX_DEBUG_NEW_TOKENS)
    print("\n===== Stepwise logits comparison (first {} steps) =====".format(steps))

    for i in range(steps):
        b = b_scores[i]
        l = l_scores[i]
        assert b.shape == l.shape, f"vocab size mismatch at step {i}: {b.shape} vs {l.shape}"

        # Baseline / Lazy 各自 argmax token
        b_id = int(torch.argmax(b).item())
        l_id = int(torch.argmax(l).item())
        b_tok = tokenizer.decode([b_id], skip_special_tokens=False)
        l_tok = tokenizer.decode([l_id], skip_special_tokens=False)

        # 当前生成的 token（实际从序列里拿）
        b_gen_id = int(b_new_tokens[i].item()) if i < len(b_new_tokens) else None
        l_gen_id = int(l_new_tokens[i].item()) if i < len(l_new_tokens) else None
        b_gen_tok = tokenizer.decode([b_gen_id], skip_special_tokens=False) if b_gen_id is not None else "<none>"
        l_gen_tok = tokenizer.decode([l_gen_id], skip_special_tokens=False) if l_gen_id is not None else "<none>"

        l2 = torch.mean((b - l) ** 2).sqrt().item()

        print(f"\n[Step {i}] L2(baseline, lazy) = {l2:.4f}")
        print(f"  Baseline argmax: id={b_id}, tok={repr(b_tok)}")
        print(f"  Lazy     argmax: id={l_id}, tok={repr(l_tok)}")
        print(f"  Baseline generated token: id={b_gen_id}, tok={repr(b_gen_tok)}")
        print(f"  Lazy     generated token: id={l_gen_id}, tok={repr(l_gen_tok)}")

        # 打印 Baseline top-5 及其在 Lazy 中的 logits
        topk = 5
        b_topv, b_topi = torch.topk(b, topk)
        l_on_b = l[b_topi]
        print("  Top-5 baseline tokens and logits (baseline vs lazy):")
        for rank in range(topk):
            tid = int(b_topi[rank].item())
            tok = tokenizer.decode([tid], skip_special_tokens=False)
            print(
                f"    {rank+1}: id={tid}, tok={repr(tok)}, "
                f"b_logit={b_topv[rank].item():.3f}, l_logit={l_on_b[rank].item():.3f}"
            )


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

    debug_steps = min(MAX_DEBUG_NEW_TOKENS, max_gen)

    b_scores, b_new_tokens, b_inputs, b_prefill_logits = run_baseline_stepwise(tokenizer, device, model_prompt, debug_steps)
    l_scores, l_new_tokens, l_inputs, l_prefill_logits = run_lazy_stepwise(tokenizer, device, model_prompt, debug_steps)

    # Prefill logits L2 对比
    assert b_prefill_logits.shape == l_prefill_logits.shape
    # 统一到 float32 + CPU 后计算 L2
    dist = torch.norm(b_prefill_logits.float() - l_prefill_logits.float(), p=2).item()
    print(f"\n>>> Prefill Logits L2 Distance: {dist:.6f}")

    compare_scores(b_scores, l_scores, tokenizer, b_new_tokens, l_new_tokens)


if __name__ == "__main__":
    main()
