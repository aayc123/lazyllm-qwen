import argparse
import json
import re
from typing import Optional


def postprocess_pred(text: str, dataset_name: Optional[str] = None) -> str:
    original = text or ""
    text = original.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1]

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip()

    for prefix in ("Answer:", "答案：", "回答："):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

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
        fallback = re.sub(r"<think>.*", "", original, flags=re.DOTALL).strip()
        text = fallback.splitlines()[0].strip() if fallback else original.strip()

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--dataset", dest="dataset", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, open(
        args.out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            row["pred"] = postprocess_pred(row.get("pred", ""), dataset_name=args.dataset)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"cleaned {n} lines -> {args.out_path}")


if __name__ == "__main__":
    main()
