#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_hotpot_yesno_jsonl.py

Create *clean* train/eval JSONL files of HotpotQA **yes/no only** examples,
with **full context**, and with **Question after Context** and **Answer included**
in the same string — ready for your `extract_documents_from_jsonl()`.

Key properties
--------------
- Reads from the SAME directory the script is located in by default.
  (Or pass --data_dir to override.)
- Uses only `hotpot_train_v1.1.json` (as requested). Dev/test are ignored.
- Filters to examples whose gold answer is exactly "yes" or "no" (case-insensitive).
- For each example, builds one document string:
      [Title_1]
      Sent_1
      Sent_2
      ...

      [Title_2]
      Sent_1
      ...

      Question: <original question>
      Answer: <yes|no>
- Removes fields the model doesn't need to *see* (e.g., qid) from the `text`.
  (We keep a sidecar labels file with doc_id→gold answer for evaluation.)
- Deterministic split into train/eval via --train_ratio and --seed.
- Outputs are written into the SAME directory as the input file:
      hotpot_yesno_train.jsonl
      hotpot_yesno_eval.jsonl
  with matching labels:
      hotpot_yesno_train.labels.jsonl
      hotpot_yesno_eval.labels.jsonl

Why JSONL?
-----------
Your `extract_documents_from_jsonl()` consumes `{"text": "..."}`
per line, treating each line as one *document*. That perfectly aligns
with your sentence-based model and downstream chunking.

Usage
-----
$ python make_hotpot_yesno_jsonl.py
# or with explicit options
$ python make_hotpot_yesno_jsonl.py --data_dir /path/to/data --train_file hotpot_train_v1.1.json --train_ratio 0.9 --seed 13

Notes
-----
- We *do not* touch/require the distractor or fullwiki test files.
- The output doc_ids used by your pipeline will be:
      jsonl_<basename>_<i>
  where <basename> is the output filename (e.g., hotpot_yesno_train.jsonl),
  and <i> is the zero-based line index. We precompute and save these doc_ids
  in the sidecar label files for convenient evaluation.
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any

def _norm_answer(a: str) -> str:
    return (a or "").strip().lower()

def _build_text(example: Dict[str, Any]) -> str:
    """
    Construct the training/eval string:
      [Title]
      sentence...

      Question: ...
      Answer: yes|no
    """
    parts: List[str] = []
    # full context — preserve given order
    for pair in example.get("context", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            # Be defensive; skip malformed
            continue
        title, sentences = pair[0], pair[1]
        title = str(title or "").strip()
        parts.append(f"[{title}]") if title else None
        if isinstance(sentences, list):
            for s in sentences:
                s = (s or "").rstrip()
                if s:
                    parts.append(s)
        parts.append("")  # blank line between paragraphs

    # question then answer
    q = (example.get("question") or "").strip()
    a = _norm_answer(example.get("answer"))
    parts.append(f"Question: {q}")
    parts.append(f"Answer: {a}")
    return "\n".join(parts).rstrip() + "\n"

def _load_hotpot_train(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure in {path}; expected a list.")
    return data

def _filter_yesno(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ex in examples:
        a = _norm_answer(ex.get("answer"))
        if a in ("yes", "no"):
            out.append(ex)
    return out

def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="Directory containing hotpot_train_v1.1.json")
    ap.add_argument("--train_file", type=str, default="hotpot_train_v1.1.json",
                    help="Filename of HotpotQA train JSON")
    ap.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio (rest is eval)")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed for deterministic split")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    train_path = os.path.join(data_dir, args.train_file)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Could not find {train_path}. Place this script in the same directory as the file "
                                f"or pass --data_dir/--train_file.")

    print(f"Reading HotpotQA: {train_path}")
    raw = _load_hotpot_train(train_path)
    yesno = _filter_yesno(raw)
    print(f"Total examples in train: {len(raw):,}")
    print(f"Yes/No examples:        {len(yesno):,}")

    # Build JSONL rows
    docs: List[Dict[str, Any]] = []
    for ex in yesno:
        text = _build_text(ex)
        # Only 'text' is used by your extractor; we keep minimal metadata for convenience
        docs.append({"text": text})

    # Deterministic split
    rng = random.Random(args.seed)
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    n_train = int(round(args.train_ratio * len(docs)))
    train_idx = indices[:n_train]
    eval_idx  = indices[n_train:]

    train_rows = [docs[i] for i in train_idx]
    eval_rows  = [docs[i] for i in eval_idx]

    # Output JSONL files (same directory as input)
    out_train = os.path.join(data_dir, "hotpot_yesno_train.jsonl")
    out_eval  = os.path.join(data_dir, "hotpot_yesno_eval.jsonl")
    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_eval,  eval_rows)
    print(f"Wrote: {out_train}  ({len(train_rows):,} lines)")
    print(f"Wrote: {out_eval}   ({len(eval_rows):,} lines)")

    # Sidecar label files: doc_id mapping for evaluation
    # Your extractor will assign: doc_id = f"jsonl_{basename}_{i}"
    base_train = os.path.basename(out_train)
    base_eval  = os.path.basename(out_eval)

    # Build the aligned labels using the *post*-shuffle ordering
    def _labels_for(split_rows, split_name: str, base_name: str, idx_list: List[int]) -> List[Dict[str, Any]]:
        labels = []
        for i, src_idx in enumerate(idx_list):
            ex = yesno[src_idx]  # original yes/no example
            gold = _norm_answer(ex.get("answer"))
            qid = ex.get("_id", "")
            labels.append({
                "doc_id": f"jsonl_{base_name}_{i}",
                "qid": qid,
                "answer": gold,
                "question": ex.get("question", ""),
            })
        return labels

    train_labels = _labels_for(train_rows, "train", base_train, train_idx)
    eval_labels  = _labels_for(eval_rows,  "eval",  base_eval,  eval_idx)

    out_train_labels = os.path.join(data_dir, "hotpot_yesno_train.labels.jsonl")
    out_eval_labels  = os.path.join(data_dir, "hotpot_yesno_eval.labels.jsonl")
    _write_jsonl(out_train_labels, train_labels)
    _write_jsonl(out_eval_labels,  eval_labels)
    print(f"Wrote: {out_train_labels}")
    print(f"Wrote: {out_eval_labels}")

    print("\nNext steps:")
    print("  • Training: point your data loader to hotpot_yesno_train.jsonl")
    print("  • Evaluation: use hotpot_yesno_eval.jsonl for inputs; compare predictions against the sidecar labels by doc_id.")
    print("  • Your pipeline’s extract_documents_from_jsonl() will treat each JSONL line as one document.")

if __name__ == "__main__":
    main()
