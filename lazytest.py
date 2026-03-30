import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import json
import time
import argparse

import torch
from datasets import load_dataset
from transformers import Qwen3ForCausalLM, AutoTokenizer
from tqdm import tqdm

from models import LazyQwen3ForCausalLM
from caches import KVCache, AuxCache


# 可按需修改
LOCAL_PATH = "/data/zn/model/models/Qwen3-8B"
LONGBENCH_DIR = "/data/zn/longbench"
DATA_FILE = os.path.join(LONGBENCH_DIR, "data/qasper.jsonl")
DATASET2PROMPT_PATH = os.path.join(LONGBENCH_DIR, "config/dataset2prompt.json")
DATASET2MAXLEN_PATH = os.path.join(LONGBENCH_DIR, "config/dataset2maxlen.json")
MAX_INPUT_LEN = 5200  # 与 example.py 保持一致的截断长度
MODEL_NAME = "lazy_qwen3"  # 对应 longbench/pred 下的子目录名


def build_prompt(sample, dataset_name: str = "qasper") -> str:
    with open(DATASET2PROMPT_PATH, "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    prompt_format = dataset2prompt[dataset_name]
    return prompt_format.format(**sample)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_longbench_qasper_lazy(max_samples: int = -1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

    with open(DATASET2MAXLEN_PATH, "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)
    max_gen = int(dataset2maxlen["qasper"])

    print("Loading qasper data...")
    data = load_dataset("json", data_files=DATA_FILE, split="train")
    num_samples_total = len(data)
    print(f"Total samples in dataset: {num_samples_total}")

    if max_samples is not None and max_samples > 0:
        num_to_run = min(max_samples, num_samples_total)
        print(f"Will only run first {num_to_run} samples (0..{num_to_run-1})")
        data_iter = (data[i] for i in range(num_to_run))
    else:
        num_to_run = num_samples_total
        print("Will run all samples")
        data_iter = iter(data)

    print("Loading base Qwen3 model (CPU, bfloat16) to get state_dict...")
    qwen3_model = Qwen3ForCausalLM.from_pretrained(
        LOCAL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    state_dict = qwen3_model.state_dict()
    config = qwen3_model.config
    del qwen3_model
    torch.cuda.empty_cache()

    print("Building LazyQwen3 model from state_dict...")
    lazy_model = LazyQwen3ForCausalLM.from_qwen3_state_dict(
        state_dict,
        config,
        pruning_rates={i: 0 for i in range(config.num_hidden_layers)},
    ).to(device)
    lazy_model.eval()
    del state_dict

    out_dir = os.path.join(LONGBENCH_DIR, "pred", MODEL_NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qasper.jsonl")
    print(f"Writing predictions to: {out_path}")

    total_prefill_ms = 0.0
    total_gen_ms = 0.0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(tqdm(data_iter, total=num_to_run, desc="qasper-lazy")):
            prompt = build_prompt(sample, "qasper")
            messages = [{"role": "user", "content": prompt}]
            model_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 仅对第一个样本打印一次 model_prompt 信息，方便与 baseline 脚本对比
            if idx == 0:
                print("\n[LazyTest] model_prompt head:", repr(model_prompt[:200]))
                encoded_dbg = tokenizer(
                    [model_prompt],
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_INPUT_LEN,
                )
                print("[LazyTest] model_prompt tokenized length:", encoded_dbg["input_ids"].shape[1])

            inputs = tokenizer(
                [model_prompt],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LEN,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            seq_len = input_ids.shape[1]
            max_length = seq_len + max_gen

            kv_cache = KVCache(
                lazy_model.config.num_hidden_layers,
                1,
                lazy_model.config.num_key_value_heads,
                max_length,
                lazy_model.config.head_dim,
                device,
                dtype=torch.bfloat16,
            )
            aux_cache = AuxCache(
                lazy_model.config.num_hidden_layers,
                1,
                max_length,
                lazy_model.config.hidden_size,
                device,
                dtype=torch.bfloat16,
            )

            with torch.no_grad():
                cache_position = torch.arange(seq_len, device=device)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

                _sync(device)
                t0 = time.time()
                _ = lazy_model(
                    kv_cache=kv_cache,
                    aux_cache=aux_cache,
                    cache_position=cache_position,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                )
                _sync(device)
                t1 = time.time()
            prefill_ms = (t1 - t0) * 1000.0
            total_prefill_ms += prefill_ms

            kv_cache = KVCache(
                lazy_model.config.num_hidden_layers,
                1,
                lazy_model.config.num_key_value_heads,
                max_length,
                lazy_model.config.head_dim,
                device,
                dtype=torch.bfloat16,
            )
            aux_cache = AuxCache(
                lazy_model.config.num_hidden_layers,
                1,
                max_length,
                lazy_model.config.hidden_size,
                device,
                dtype=torch.bfloat16,
            )

            with torch.no_grad():
                _sync(device)
                t2 = time.time()
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
                _sync(device)
                t3 = time.time()
            gen_ms = (t3 - t2) * 1000.0
            total_gen_ms += gen_ms

            pred = tokenizer.decode(
                output_ids[0][seq_len:],
                skip_special_tokens=True,
            )

            # 调试：打印每条样本的前 200 个字符，核对与 jsonl 中是否一致
            print(f"[Lazy qasper] sample {idx}, pred head: {pred[:200]!r}")

            record = {
                "pred": pred,
                "answers": sample.get("answers"),
                "all_classes": sample.get("all_classes"),
                "length": sample.get("length"),
                "prefill_ms": prefill_ms,
                "gen_ms": gen_ms,
            }
            json.dump(record, f_out, ensure_ascii=False)
            f_out.write("\n")

    avg_prefill = total_prefill_ms / max(num_to_run, 1)
    avg_gen = total_gen_ms / max(num_to_run, 1)
    print(f"Average prefill time: {avg_prefill:.1f} ms / sample (over {num_to_run} samples)")
    print(f"Average generate time: {avg_gen:.1f} ms / sample (over {num_to_run} samples)")
    print(f"Predictions written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="只跑前 N 条样本，默认 -1 表示跑全量",
    )
    args = parser.parse_args()

    run_longbench_qasper_lazy(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
