import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import json
import time

import torch
from datasets import load_dataset
from transformers import Qwen3ForCausalLM, AutoTokenizer
from tqdm import tqdm


# 可按需修改
LOCAL_PATH = "/data/zn/model/models/Qwen3-8B"
LONGBENCH_DIR = "/data/zn/longbench"
DATA_FILE = os.path.join(LONGBENCH_DIR, "data/qasper.jsonl")
DATASET2PROMPT_PATH = os.path.join(LONGBENCH_DIR, "config/dataset2prompt.json")
DATASET2MAXLEN_PATH = os.path.join(LONGBENCH_DIR, "config/dataset2maxlen.json")
MAX_INPUT_LEN = 5200  # 与 example.py 保持一致的截断长度
MODEL_NAME = "qwen3-8b-baseline"  # 对应 longbench/pred 下的子目录名


def build_prompt(sample, dataset_name: str = "qasper") -> str:
    with open(DATASET2PROMPT_PATH, "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    prompt_format = dataset2prompt[dataset_name]
    return prompt_format.format(**sample)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_longbench_qasper_baseline():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

    with open(DATASET2MAXLEN_PATH, "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)
    max_gen = int(dataset2maxlen["qasper"])

    print("Loading qasper data...")
    data = load_dataset("json", data_files=DATA_FILE, split="train")
    num_samples = len(data)
    print(f"Total samples: {num_samples}")

    print("Loading baseline Qwen3 model...")
    model = Qwen3ForCausalLM.from_pretrained(
        LOCAL_PATH,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    out_dir = os.path.join(LONGBENCH_DIR, "pred", MODEL_NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qasper.jsonl")
    print(f"Writing predictions to: {out_path}")

    total_prefill_ms = 0.0
    total_gen_ms = 0.0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(tqdm(data, desc="qasper")):
            prompt = build_prompt(sample, "qasper")
            messages = [{"role": "user", "content": prompt}]
            model_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(
                [model_prompt],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LEN,
            ).to(device)

            # 计时：prefill（单次 forward，不参与 generate）
            with torch.no_grad():
                _sync(device)
                t0 = time.time()
                _ = model(**inputs)
                _sync(device)
                t1 = time.time()
            prefill_ms = (t1 - t0) * 1000.0
            total_prefill_ms += prefill_ms

            # 计时：完整生成（包含 prefill+decode）
            with torch.no_grad():
                _sync(device)
                t2 = time.time()
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_gen,
                    do_sample=False,
                )
                _sync(device)
                t3 = time.time()
            gen_ms = (t3 - t2) * 1000.0
            total_gen_ms += gen_ms

            pred = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

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

    avg_prefill = total_prefill_ms / max(num_samples, 1)
    avg_gen = total_gen_ms / max(num_samples, 1)
    print(f"Average prefill time: {avg_prefill:.1f} ms / sample")
    print(f"Average generate time: {avg_gen:.1f} ms / sample")


def main():
    run_longbench_qasper_baseline()


if __name__ == "__main__":
    main()
