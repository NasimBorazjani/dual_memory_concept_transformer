#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_binding_gpt2.py

Behavioral evaluation for a plain GPT‑2 baseline trained with your GPT‑2
training pipeline. Functionally mirrors your memory‑model evaluation
(evaluate_binding.py), with the same tasks, metrics, logs and file formats.

Tasks:
  (1) Object–feature attribution (paper-style)
  (2) Father–son "reversal curse"

Protocol (GPT‑2):
  • Build a single prefix:  S1 + "\n" + query_prefix   (query_prefix ends with a space)
  • Run GPT‑2 once; read logits at the last prefix position (next-token distribution).

Outputs in --out_dir:
  • config_detected.txt
  • object_feature_samples.txt / .jsonl / object_feature_summary.json
  • fatherson_samples.txt / .jsonl / fatherson_summary.json

CLI:
  --run_dir       Path to a single baseline run (with checkpoints/best_model.pt)
  --lexicon_dir   Folder containing colors.txt, sizes.txt, objects.txt, male_names.txt
  --out_dir       Output directory
  --n_per_condition, --topk, --candidate_topk, --print_every, --seed

Dependencies:
  • torch, transformers, tqdm, numpy
"""

import os, re, json, sys, math, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm


# ============================== Utilities ==============================

def _read_text_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                out.append(w)
    return out

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def _ensure_trailing_space(prefix: str) -> str:
    return prefix if prefix.endswith(" ") else prefix + " "

def _decode_ids(tok: GPT2Tokenizer, ids: List[int]) -> List[str]:
    return [tok.decode([i], clean_up_tokenization_spaces=False) for i in ids]


# ============================== Tokenizer ==============================

def build_stock_tokenizer() -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    # GPT‑2 typically has no pad token; we don't pad here but ensure decode/encode is standard.
    return tok

def tok_id_for_answer(tok: GPT2Tokenizer, word: str) -> Optional[int]:
    """
    Return the single-token id for ' word' (leading space) if it encodes to exactly 1 token; else None.
    """
    ids = tok.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None

def filter_single_token_words(tok: GPT2Tokenizer, words: List[str]) -> List[str]:
    out = []
    for w in words:
        if tok_id_for_answer(tok, w) is not None:
            out.append(w)
    return out


# ============================== Model loading ==============================

def _try_parse_config_from_logs(log_text: str) -> Optional[Dict[str, Any]]:
    """
    The baseline trainer logs: "Starting GPT‑2 baseline with configuration:\n{ ...JSON... }"
    Try to extract the first JSON block that follows.
    """
    marker = "Starting GPT‑2 baseline with configuration:"
    i = log_text.find(marker)
    if i < 0:
        return None
    j = log_text.find("{", i)
    if j < 0:
        return None
    # Heuristic: find a closing brace; expand cautiously
    k = j
    while k < len(log_text):
        k = log_text.find("}", k + 1)
        if k < 0:
            break
        blob = log_text[j:k+1]
        try:
            return json.loads(blob)
        except Exception:
            continue
    return None

@dataclass
class DiscoveredConfig:
    args: Dict[str, Any]
    tokenizer: GPT2Tokenizer
    model_config: Any  # transformers config
    device_str: str

def load_gpt2_from_run_dir(run_dir: str, device: Optional[str] = None) -> Tuple[GPT2LMHeadModel, DiscoveredConfig]:
    """
    Load a GPT‑2 LMHead model and tokenizer from a baseline run. Expects:
      run_dir/checkpoints/best_model.pt
    Will try to glean args from logs if present; otherwise falls back to model.config only.
    """
    run = Path(run_dir)
    ckpt_best = run / "checkpoints" / "best_model.pt"
    ckpt_alt  = run / "checkpoints" / "model.pt"   # optional fallback

    if ckpt_best.exists():
        ckpt_path = ckpt_best
    elif ckpt_alt.exists():
        ckpt_path = ckpt_alt
    else:
        raise FileNotFoundError(f"Could not find checkpoint under {run}/checkpoints (expected 'best_model.pt').")

    tok = build_stock_tokenizer()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    state = torch.load(ckpt_path, map_location=dev)
    # Load strict; if your checkpoint is a plain state_dict from .state_dict() this will work.
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()

    # Try to recover training args (best‑effort)
    args_json: Dict[str, Any] = {}
    logs_dir = run / "logs"
    candidates = ["experiment.log", "train.log", "run.log"]
    for name in candidates:
        p = logs_dir / name
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                parsed = _try_parse_config_from_logs(txt)
                if parsed:
                    args_json = parsed
                    break
            except Exception:
                pass

    disc = DiscoveredConfig(args=args_json, tokenizer=tok, model_config=model.config, device_str=device)
    return model, disc


# ============================== Two-sentence scoring (GPT‑2) ==============================

@torch.no_grad()
def next_token_logits_two_sentence_gpt2(
    model: GPT2LMHeadModel,
    tok: GPT2Tokenizer,
    sentence1: str,
    query_prefix: str,
) -> torch.Tensor:
    """
    Build a single prefix "sentence1 + \\n + query_prefix" (with trailing space)
    and return logits for the NEXT token at the last prefix position: shape [V] on CPU.
    """
    device = next(model.parameters()).device
    qpref = _ensure_trailing_space(query_prefix)
    prefix = sentence1.rstrip() + "\n" + qpref
    ids = tok.encode(prefix, add_special_tokens=False)
    if len(ids) == 0:
        # If somehow empty, still make a call with minimal input.
        ids = tok.encode(" ", add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=inp)
    # logits shape: [B=1, T, V]; the distribution for the NEXT token is logits at position T-1.
    return out.logits[0, -1, :].detach().cpu()


# ============================== Metrics helpers ==============================

@dataclass
class MetricOptions:
    topk: int = 5
    candidate_topk: int = 1

def compute_metrics_from_logits(
    logits: torch.Tensor,
    target_id: int,
    candidate_ids: List[int],
    opts: MetricOptions,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - top1_global (bool)
      - topk_global (bool for k=opts.topk)
      - cand_max (bool)                  # highest among candidates
      - cand_topk (bool, k=opts.candidate_topk)
      - global_rank (int, 1=best)
      - candidate_rank (int within candidates, 1=best)
      - target_prob (float), candidate_probs (dict)
    """
    lp = logits.cpu().numpy()
    probs = _softmax_np(lp)

    # global rank of target
    order = np.argsort(-probs)  # descending
    # find first index equal to target_id
    where = np.nonzero(order == target_id)[0]
    vocab_rank = int(where[0]) + 1 if where.size > 0 else len(probs)

    top1_global = (vocab_rank == 1)
    topk_global = (vocab_rank <= max(1, opts.topk))

    # candidate‑restricted ranks
    cand_probs = {cid: probs[cid] for cid in candidate_ids}
    cand_sorted = sorted(candidate_ids, key=lambda i: -cand_probs[i])
    cand_rank = int(cand_sorted.index(target_id)) + 1 if target_id in cand_probs else len(candidate_ids) + 1
    cand_max = (cand_rank == 1)
    cand_topk = (cand_rank <= max(1, opts.candidate_topk))

    return {
        "top1_global": bool(top1_global),
        "topk_global": bool(topk_global),
        "cand_max": bool(cand_max),
        "cand_topk": bool(cand_topk),
        "global_rank": vocab_rank,
        "candidate_rank": cand_rank,
        "target_prob": float(probs[target_id]),
        "candidate_probs": {int(i): float(cand_probs[i]) for i in candidate_ids},
        "top_tokens": None,  # filled by caller if needed
    }

def topk_tokens_from_logits(tok: GPT2Tokenizer, logits: torch.Tensor, k: int = 5) -> List[Tuple[int, str, float]]:
    lp = logits.cpu().numpy()
    probs = _softmax_np(lp)
    idx = np.argsort(-probs)[:k]
    return [(int(i), tok.decode([int(i)], clean_up_tokenization_spaces=False), float(probs[i])) for i in idx]


# ============================== Object–feature task ==============================

@dataclass
class ObjFeatSample:
    obj1: str; color1: str; size1: str
    obj2: str; color2: str; size2: str
    query_feature: str  # "color" or "size"
    query_obj_idx: int  # 1 or 2
    order_color_first: bool  # True: "color and size", False: "size and color"
    reversed_object_order: bool  # positional sensitivity / swap order

def build_objfeat_sentence1(s: ObjFeatSample) -> str:
    # order of features inside each object's phrase
    if s.order_color_first:
        f1 = f"{s.color1} and {s.size1}"
        f2 = f"{s.color2} and {s.size2}"
    else:
        f1 = f"{s.size1} and {s.color1}"
        f2 = f"{s.size2} and {s.color2}"
    if not s.reversed_object_order:
        return f"The {s.obj1} is {f1} and the {s.obj2} is {f2}."
    else:
        return f"The {s.obj2} is {f2} and the {s.obj1} is {f1}."

def build_objfeat_query_prefix(s: ObjFeatSample) -> str:
    obj = s.obj1 if s.query_obj_idx == 1 else s.obj2
    return _ensure_trailing_space(f"So, the {s.query_feature} of the {obj} is")

def objfeat_target_and_candidates(tok: GPT2Tokenizer, s: ObjFeatSample) -> Tuple[int, List[int], Dict[str,int]]:
    target_word = (s.color1 if s.query_obj_idx == 1 else s.color2) if s.query_feature == "color" \
                  else (s.size1  if s.query_obj_idx == 1 else s.size2)
    target_id = tok_id_for_answer(tok, target_word)
    cand_words = [s.color1, s.size1, s.color2, s.size2]  # four in‑context features
    cand_ids = []
    for w in cand_words:
        tid = tok_id_for_answer(tok, w)
        if tid is None:
            continue
        cand_ids.append(tid)
    return target_id, cand_ids, {w: tok_id_for_answer(tok, w) for w in cand_words}

def sample_objfeat_instances(
    rng: np.random.Generator,
    objects: List[str], colors: List[str], sizes: List[str],
    n_per_condition: int
) -> List[ObjFeatSample]:
    """
    Build a balanced list covering (feature × object × order) for both normal and reversed conditions.
    """
    out: List[ObjFeatSample] = []
    feats = ["color", "size"]
    obj_idx = [1, 2]
    orders = [True, False]  # color-first vs size-first
    combos = [(f, oi, od) for f in feats for oi in obj_idx for od in orders]
    per_combo = max(1, n_per_condition // len(combos))

    def pick_two(xs: List[str]) -> Tuple[str, str]:
        a = xs[int(rng.integers(0, len(xs)))]
        b = a
        while b == a:
            b = xs[int(rng.integers(0, len(xs)))]
        return a, b

    # normal
    for (f, oi, od) in combos:
        for _ in range(per_combo):
            o1, o2 = pick_two(objects)
            c1, c2 = pick_two(colors)
            z1, z2 = pick_two(sizes)
            out.append(ObjFeatSample(
                obj1=o1, color1=c1, size1=z1,
                obj2=o2, color2=c2, size2=z2,
                query_feature=f, query_obj_idx=oi,
                order_color_first=od,
                reversed_object_order=False
            ))
    # reversed
    for (f, oi, od) in combos:
        for _ in range(per_combo):
            o1, o2 = pick_two(objects)
            c1, c2 = pick_two(colors)
            z1, z2 = pick_two(sizes)
            out.append(ObjFeatSample(
                obj1=o1, color1=c1, size1=z1,
                obj2=o2, color2=c2, size2=z2,
                query_feature=f, query_obj_idx=oi,
                order_color_first=od,
                reversed_object_order=True
            ))
    # pad if rounding short
    def count_cond(cond: bool) -> int:
        return sum(1 for s in out if s.reversed_object_order == cond)
    while count_cond(False) < n_per_condition:
        out.append(next(s for s in out if not s.reversed_object_order))
    while count_cond(True) < n_per_condition:
        out.append(next(s for s in out if s.reversed_object_order))
    rng.shuffle(out)
    return out


# ============================== Father–son task ==============================

@dataclass
class FatherSonSample:
    father: str
    son: str
    reversed_context: bool  # True => "{S} is the son of {F}."

def build_fatherson_sentence1(s: FatherSonSample) -> str:
    if s.reversed_context:
        return f"{s.son} is the son of {s.father}."
    return f"{s.father} is the father of {s.son}."

def build_fatherson_query_prefix(s: FatherSonSample) -> str:
    return _ensure_trailing_space(f"So, {s.son} is the son of")

def fatherson_target_and_candidates(tok: GPT2Tokenizer, s: FatherSonSample) -> Tuple[int, List[int], Dict[str,int]]:
    target_id = tok_id_for_answer(tok, s.father)
    cand = [s.father, s.son]
    cand_ids = [tok_id_for_answer(tok, w) for w in cand]
    return target_id, cand_ids, {w: tok_id_for_answer(tok, w) for w in cand}

def sample_fatherson_instances(rng: np.random.Generator, male_names: List[str], n_per_condition: int) -> List[FatherSonSample]:
    out: List[FatherSonSample] = []
    def pick_two(xs: List[str]) -> Tuple[str, str]:
        a = xs[int(rng.integers(0, len(xs)))]
        b = a
        while b == a:
            b = xs[int(rng.integers(0, len(xs)))]
        return a, b
    # normal
    for _ in range(n_per_condition):
        f, s = pick_two(male_names)
        out.append(FatherSonSample(father=f, son=s, reversed_context=False))
    # reversed
    for _ in range(n_per_condition):
        f, s = pick_two(male_names)
        out.append(FatherSonSample(father=f, son=s, reversed_context=True))
    rng.shuffle(out)
    return out


# ============================== Logging header ==============================

def header_text_from_discovered(run_dir: str, disc: DiscoveredConfig, args: argparse.Namespace) -> str:
    cfg = disc.model_config
    lines = []
    lines.append("="*88)
    lines.append("MODEL CONFIG DISCOVERED FROM RUN DIRECTORY (GPT‑2 baseline)")
    lines.append("="*88)
    lines.append(f"run_dir: {run_dir}")
    lines.append(f"device: {disc.device_str}")
    lines.append(f"model_name_or_type: gpt2")
    lines.append(f"n_layer={getattr(cfg, 'n_layer', 'NA')}  n_head={getattr(cfg, 'n_head', 'NA')}  n_embd={getattr(cfg, 'n_embd', 'NA')}")
    lines.append(f"n_positions={getattr(cfg, 'n_positions', 'NA')}  n_ctx={getattr(cfg, 'n_ctx', 'NA')}")
    if disc.args:
        lines.append("")
        lines.append("TRAINING ARGS (best‑effort parse from logs):")
        try:
            lines.append(json.dumps(disc.args, indent=2))
        except Exception:
            lines.append("(unavailable)")
    lines.append("")
    lines.append("="*88)
    lines.append("EVALUATION SETTINGS")
    lines.append("="*88)
    lines.append(f"n_per_condition={args.n_per_condition}  topk={args.topk}  candidate_topk={args.candidate_topk}")
    lines.append(f"print_every={args.print_every}")
    lines.append("")
    return "\n".join(lines)


# ============================== Evaluation loops ==============================

def run_object_feature_eval(
    model: GPT2LMHeadModel,
    disc: DiscoveredConfig,
    objects: List[str], colors: List[str], sizes: List[str],
    out_dir: Path, n_per_condition: int, print_every: int,
    opts: MetricOptions, seed: int
):
    tok = disc.tokenizer
    rng = np.random.default_rng(seed)

    samples = sample_objfeat_instances(rng, objects, colors, sizes, n_per_condition)

    txt_path = out_dir / "object_feature_samples.txt"
    jsonl_path = out_dir / "object_feature_samples.jsonl"
    summary_path = out_dir / "object_feature_summary.json"

    with txt_path.open("w", encoding="utf-8") as ftxt, jsonl_path.open("w", encoding="utf-8") as fj:
        total = len(samples)
        pbar = tqdm(total=total, desc="Object–Feature eval")

        agg = {
            "normal":   {"top1_global":0, "cand_max":0, "topk_global":0, "cand_topk":0, "count":0,
                         "mean_global_rank":[], "mean_candidate_rank":[]},
            "reversed": {"top1_global":0, "cand_max":0, "topk_global":0, "cand_topk":0, "count":0,
                         "mean_global_rank":[], "mean_candidate_rank":[]},
        }

        for i, s in enumerate(samples, 1):
            sent1 = build_objfeat_sentence1(s)
            qpref = build_objfeat_query_prefix(s)
            logits = next_token_logits_two_sentence_gpt2(model, tok, sent1, qpref)

            target_id, cand_ids, _cand_map = objfeat_target_and_candidates(tok, s)
            if target_id is None or any(cid is None for cid in cand_ids):
                pbar.update(1); continue

            stats = compute_metrics_from_logits(logits, target_id, cand_ids, opts)
            stats["top_tokens"] = topk_tokens_from_logits(tok, logits, k=max(5, opts.topk))

            cond = "reversed" if s.reversed_object_order else "normal"
            agg[cond]["count"] += 1
            agg[cond]["top1_global"] += int(stats["top1_global"])
            agg[cond]["topk_global"] += int(stats["topk_global"])
            agg[cond]["cand_max"]    += int(stats["cand_max"])
            agg[cond]["cand_topk"]   += int(stats["cand_topk"])
            agg[cond]["mean_global_rank"].append(stats["global_rank"])
            agg[cond]["mean_candidate_rank"].append(stats["candidate_rank"])

            if (i % max(1, print_every)) == 0:
                top_str = ", ".join([f"{w}:{p:.3f}" for _, w, p in stats["top_tokens"]])
                msg = (f"[ObjFeat {i}/{total}] cond={cond} | S1='{sent1}' | Q='{qpref}__?'\n"
                       f"  target={tok.decode([target_id], clean_up_tokenization_spaces=False).strip()} "
                       f"({stats['target_prob']:.3f}) | cand_max={stats['cand_max']} | "
                       f"top1_global={stats['top1_global']}\n"
                       f"  top-{max(5,opts.topk)}: {top_str}")
                tqdm.write(msg)
                ftxt.write(msg + "\n")

            rec = {
                "condition": cond,
                "sentence1": sent1,
                "query_prefix": qpref,
                "query_feature": s.query_feature,
                "query_obj_idx": s.query_obj_idx,
                "order_color_first": s.order_color_first,
                "reversed_object_order": s.reversed_object_order,
                "target_id": int(target_id),
                "target_str": tok.decode([target_id], clean_up_tokenization_spaces=False),
                "candidate_ids": [int(x) for x in cand_ids],
                "candidate_strs": _decode_ids(tok, cand_ids),
                "metrics": {k: (int(v) if isinstance(v, bool) else v)
                            for k, v in stats.items() if k not in ("candidate_probs","top_tokens")},
                "candidate_probs": {tok.decode([i], clean_up_tokenization_spaces=False): p
                                    for i,p in stats["candidate_probs"].items()},
                "top_tokens": [{"id":i,"str":sstr,"prob":p} for (i,sstr,p) in stats["top_tokens"]],
            }
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pbar.update(1)
        pbar.close()

        summary = {}
        for cond, d in agg.items():
            n = max(1, d["count"])
            summary[cond] = {
                "count": d["count"],
                "acc_top1_global": d["top1_global"]/n,
                "acc_topk_global": d["topk_global"]/n,
                "acc_candidate_max": d["cand_max"]/n,
                "acc_candidate_topk": d["cand_topk"]/n,
                "mean_global_rank": float(np.mean(d["mean_global_rank"])) if d["mean_global_rank"] else None,
                "mean_candidate_rank": float(np.mean(d["mean_candidate_rank"])) if d["mean_candidate_rank"] else None,
            }
        with summary_path.open("w", encoding="utf-8") as fs:
            json.dump(summary, fs, indent=2)
    return str(summary_path)

def run_fatherson_eval(
    model: GPT2LMHeadModel,
    disc: DiscoveredConfig,
    male_names: List[str],
    out_dir: Path, n_per_condition: int, print_every: int,
    opts: MetricOptions, seed: int
):
    tok = disc.tokenizer
    rng = np.random.default_rng(seed + 13)

    samples = sample_fatherson_instances(rng, male_names, n_per_condition)

    txt_path = out_dir / "fatherson_samples.txt"
    jsonl_path = out_dir / "fatherson_samples.jsonl"
    summary_path = out_dir / "fatherson_summary.json"

    with txt_path.open("w", encoding="utf-8") as ftxt, jsonl_path.open("w", encoding="utf-8") as fj:
        total = len(samples)
        pbar = tqdm(total=total, desc="Father–Son eval")

        agg = {
            "normal":   {"top1_global":0, "cand_max":0, "topk_global":0, "cand_topk":0, "count":0,
                         "mean_global_rank":[], "mean_candidate_rank":[]},
            "reversed": {"top1_global":0, "cand_max":0, "topk_global":0, "cand_topk":0, "count":0,
                         "mean_global_rank":[], "mean_candidate_rank":[]},
        }

        for i, s in enumerate(samples, 1):
            sent1 = build_fatherson_sentence1(s)
            qpref = build_fatherson_query_prefix(s)
            logits = next_token_logits_two_sentence_gpt2(model, tok, sent1, qpref)

            target_id, cand_ids, _cand_map = fatherson_target_and_candidates(tok, s)
            if target_id is None or any(cid is None for cid in cand_ids):
                pbar.update(1); continue

            stats = compute_metrics_from_logits(logits, target_id, cand_ids, opts)
            stats["top_tokens"] = topk_tokens_from_logits(tok, logits, k=max(5, opts.topk))

            cond = "reversed" if not s.reversed_context else "normal"
            agg[cond]["count"] += 1
            agg[cond]["top1_global"] += int(stats["top1_global"])
            agg[cond]["topk_global"] += int(stats["topk_global"])
            agg[cond]["cand_max"]    += int(stats["cand_max"])
            agg[cond]["cand_topk"]   += int(stats["cand_topk"])
            agg[cond]["mean_global_rank"].append(stats["global_rank"])
            agg[cond]["mean_candidate_rank"].append(stats["candidate_rank"])

            if (i % max(1, print_every)) == 0:
                top_str = ", ".join([f"{w}:{p:.3f}" for _, w, p in stats["top_tokens"]])
                msg = (f"[FatherSon {i}/{total}] cond={cond} | S1='{sent1}' | Q='{qpref}__?'\n"
                       f"  target={tok.decode([target_id], clean_up_tokenization_spaces=False).strip()} "
                       f"({stats['target_prob']:.3f}) | cand_max={stats['cand_max']} | "
                       f"top1_global={stats['top1_global']}\n"
                       f"  top-{max(5,opts.topk)}: {top_str}")
                tqdm.write(msg)
                ftxt.write(msg + "\n")

            rec = {
                "condition": cond,
                "sentence1": sent1,
                "query_prefix": qpref,
                "target_id": int(target_id),
                "target_str": tok.decode([target_id], clean_up_tokenization_spaces=False),
                "candidate_ids": [int(x) for x in cand_ids],
                "candidate_strs": _decode_ids(tok, cand_ids),
                "metrics": {k: (int(v) if isinstance(v, bool) else v)
                            for k, v in stats.items() if k not in ("candidate_probs","top_tokens")},
                "candidate_probs": {tok.decode([i], clean_up_tokenization_spaces=False): p
                                    for i,p in stats["candidate_probs"].items()},
                "top_tokens": [{"id":i,"str":sstr,"prob":p} for (i,sstr,p) in stats["top_tokens"]],
            }
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pbar.update(1)
        pbar.close()

        summary = {}
        for cond, d in agg.items():
            n = max(1, d["count"])
            summary[cond] = {
                "count": d["count"],
                "acc_top1_global": d["top1_global"]/n,
                "acc_topk_global": d["topk_global"]/n,
                "acc_candidate_max": d["cand_max"]/n,      # identical to “correct of two”
                "acc_candidate_topk": d["cand_topk"]/n,
                "mean_global_rank": float(np.mean(d["mean_global_rank"])) if d["mean_global_rank"] else None,
                "mean_candidate_rank": float(np.mean(d["mean_candidate_rank"])) if d["mean_candidate_rank"] else None,
            }
        with summary_path.open("w", encoding="utf-8") as fs:
            json.dump(summary, fs, indent=2)
    return str(summary_path)


# ============================== CLI ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Baseline run directory (produced by your GPT‑2 trainer)")
    ap.add_argument("--lexicon_dir", type=str, required=True, help="Directory with colors.txt, sizes.txt, objects.txt, male_names.txt")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write logs and summaries")
    ap.add_argument("--n_per_condition", type=int, default=100, help="Prompts per condition (normal and reversed) for each task")
    ap.add_argument("--topk", type=int, default=5, help="Global top‑k success (target in top‑k of full vocab)")
    ap.add_argument("--candidate_topk", type=int, default=2, help="Candidate‑restricted top‑k success (e.g., 1=paper metric)")
    ap.add_argument("--print_every", type=int, default=10, help="Print a sample every N items")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load model & tokenizer
    model, disc = load_gpt2_from_run_dir(args.run_dir)
    header = header_text_from_discovered(args.run_dir, disc, args)
    with (out_dir / "config_detected.txt").open("w", encoding="utf-8") as f:
        f.write(header + "\n")

    # Lexicons (+ single‑token filtering under stock GPT‑2)
    tok = disc.tokenizer
    lex_dir = Path(args.lexicon_dir)
    objects = filter_single_token_words(tok, _read_text_file(lex_dir / "objects.txt"))
    colors  = filter_single_token_words(tok, _read_text_file(lex_dir / "colors.txt"))
    sizes   = filter_single_token_words(tok, _read_text_file(lex_dir / "sizes.txt"))
    male    = filter_single_token_words(tok, _read_text_file(lex_dir / "male_names.txt"))

    # Sanity checks
    if len(objects) < 2 or len(colors) < 2 or len(sizes) < 2:
        raise ValueError("Not enough single‑token objects/colors/sizes after filtering.")
    if len(male) < 2:
        raise ValueError("Not enough single‑token male names after filtering.")

    opts = MetricOptions(topk=max(1, args.topk), candidate_topk=max(1, args.candidate_topk))

    # Write header atop sample logs, too
    for name in ["object_feature_samples.txt", "fatherson_samples.txt"]:
        with (out_dir / name).open("w", encoding="utf-8") as f:
            f.write(header + "\n")

    # Run both tasks
    obj_summary_path = run_object_feature_eval(
        model, disc, objects, colors, sizes, out_dir,
        n_per_condition=args.n_per_condition, print_every=args.print_every, opts=opts, seed=args.seed
    )
    fs_summary_path = run_fatherson_eval(
        model, disc, male, out_dir,
        n_per_condition=args.n_per_condition, print_every=args.print_every, opts=opts, seed=args.seed
    )

    # Short console recap
    print("\nDone. Summaries:")
    print(f"  Object–feature: {obj_summary_path}")
    print(f"  Father–son:     {fs_summary_path}")


if __name__ == "__main__":
    main()
