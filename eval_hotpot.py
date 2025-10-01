#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HotpotQA evaluation (read-only) for SentenceTransformer + STM/LTM.

Fixes included (compared to your previous evaluator):
  • Correct candidate-capacity calculation: use full capacity (L-2) rather than the probe's temporary EOS position.
  • Encode the question as its own sentence (with [EOS][S_REP]) so its S_REP is inserted into STM before the answer sentence.
  • Score exactly the candidate token span (no off-by-one), and improve debug fields to reflect the true layout.

Modes:
  • --mode {span,score,auto}
    - span: enumerate candidate spans from context, score each by mean log-prob
    - score: yes/no scoring (mean log-prob), returns argmax
    - auto: 3-way decision by comparing the best span score vs max(yes,no)

Other features:
  • Robust config loading: read checkpoints/<ckpt_name>/meta.json first (if it contains 'args');
    otherwise fall back to scraping logs/experiment.log (last args JSON block)
  • Optional overrides for memory/query behavior and max_sentence_tokens
  • tqdm progress bar + random sub-sampling (--sample_n, --sample_seed)
  • Per-run folder with predictions.json, summary.json, and a periodic log.txt
    that dumps the model-visible context and the full decision trace
"""

import os
import re
import json
import math
import argparse
import logging
import random
from typing import List, Dict, Any, Tuple, Optional

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# --- your model code (imported, unmodified) ---
from model import (
    SentenceTransformerConfig,
    SentenceTransformer,
    LongTermMemoryGPU,
    DocumentLongTermMemory,
)

# --- official Hotpot evaluator (read-only) ---
import importlib.util

log = logging.getLogger("hotpot_eval")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

SPECIALS = {
    "pad_token": "[PAD]",
    "eos_token": "[EOS]",
    "srep_token": "[S_REP]",
}

# -------------------------------
# Utilities: load training config
# -------------------------------

def _load_train_args_from_log(exp_dir: str) -> Dict[str, Any]:
    log_path = os.path.join(exp_dir, "logs", "experiment.log")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Could not find experiment log at: {log_path}")
    with open(log_path, "r", encoding="utf-8") as f:
        buf = f.read()
    matches = list(re.finditer(r"\{\s*\"data_folder\".*?\}\s*", buf, flags=re.S))
    if not matches:
        raise RuntimeError("Could not find the args JSON in experiment.log.")
    raw = matches[-1].group(0)
    return json.loads(raw)


def _load_train_args(exp_dir: str, ckpt_name: str) -> Dict[str, Any]:
    ckpt_dir = os.path.join(exp_dir, "checkpoints", ckpt_name)
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and "args" in meta and isinstance(meta["args"], dict):
                return meta["args"]
        except Exception:
            pass
    return _load_train_args_from_log(exp_dir)


def _build_tokenizer(args: Dict[str, Any]) -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({
        "pad_token": SPECIALS["pad_token"],
        "eos_token": SPECIALS["eos_token"],
        "additional_special_tokens": [SPECIALS["srep_token"]],
    })
    return tok


def _cfg_from_args(args: Dict[str, Any], tokenizer: GPT2Tokenizer) -> SentenceTransformerConfig:
    def get(k, default):
        return args.get(k, default)

    return SentenceTransformerConfig(
        d_model=get("d_model", 768),
        n_heads=get("n_heads", 12),
        n_layers=get("n_layers", 12),
        ffw_mult=4,
        max_position_embeddings=get("context_length", 66),
        vocab_size=len(tokenizer),
        srep_dim=get("srep_dim", 1024),
        srep_head_depth=get("srep_head_depth", 1),
        srep_head_mult=get("srep_head_mult", 1),
        srep_head_activation=get("srep_head_activation", "gelu"),
        use_attentive_pool=get("use_attentive_pool", False),
        attn_pool_n_queries=get("attn_pool_n_queries", 1),
        max_sentences_in_short_term=get("max_sentences_in_short_term", 15),
        use_long_term_memory=get("use_long_term_memory", False),
        no_ltm_for_first_k_blocks=get("no_ltm_for_first_k_blocks", 4),
        ltm_top_k=get("ltm_top_k", 5),
        ltm_min_sim=get("ltm_min_sim", 0.3),
        ltm_min_matches=get("ltm_min_matches", 2),
        ltm_query_mode=get("ltm_query_mode", "hybrid"),
        use_stm_positional=get("use_stm_positional", True),
        stm_positional_weight=get("stm_positional_weight", 1.0),
        memory_gate_init=get("memory_gate_init", 1.0),
        dropout=get("dropout", 0.1),
        attn_dropout=get("attn_dropout", 0.1),
        srep_dropout=get("srep_dropout", 0.1),
        srep_norm_target=get("srep_norm_target", 1.0),
        srep_norm_margin=get("srep_norm_margin", 0.1),
        srep_norm_reg_weight=get("srep_norm_reg_weight", 0.01),
        debug_no_memory=get("debug_no_memory", False),
        debug_stm_only=get("debug_stm_only", False),
        max_sentence_tokens=get("max_sentence_tokens", 64),
    )


def _apply_overrides(cfg: SentenceTransformerConfig, cli: argparse.Namespace) -> SentenceTransformerConfig:
    if cli.override_use_ltm is not None:
        cfg.use_long_term_memory = bool(cli.override_use_ltm)
    if cli.override_debug_no_memory is not None:
        cfg.debug_no_memory = bool(cli.override_debug_no_memory)
    if cli.override_debug_stm_only is not None:
        cfg.debug_stm_only = bool(cli.override_debug_stm_only)
    if cli.override_ltm_query_mode:
        cfg.ltm_query_mode = cli.override_ltm_query_mode
    if cli.override_ltm_top_k is not None:
        cfg.ltm_top_k = int(cli.override_ltm_top_k)
    if cli.override_ltm_min_sim is not None:
        cfg.ltm_min_sim = float(cli.override_ltm_min_sim)
    if cli.override_ltm_min_matches is not None:
        cfg.ltm_min_matches = int(cli.override_ltm_min_matches)
    if cli.override_max_sentence_tokens is not None:
        cfg.max_position_embeddings = int(cli.override_max_sentence_tokens) + 2
        cfg.max_sentence_tokens = int(cli.override_max_sentence_tokens)
    return cfg


def _load_hotpot_module(hotpot_repo_root: str):
    p = os.path.join(hotpot_repo_root, "hotpot_evaluate_v1.py")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Could not find hotpot_evaluate_v1.py at {p}.")
    spec = importlib.util.spec_from_file_location("hotpot_eval_official", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------
# Data loader
# ---------------------------------

def load_hotpot_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------
# Build a model-ready "document" (context-only streaming)
# -------------------------------------------------------

def pack_sentences_like_training(
    sentences: List[str],
    tokenizer: GPT2Tokenizer,
    max_sentence_tokens: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    eos_id = tokenizer.convert_tokens_to_ids(SPECIALS["eos_token"])
    srep_id = tokenizer.convert_tokens_to_ids(SPECIALS["srep_token"])
    pad_id = tokenizer.pad_token_id
    L = max_sentence_tokens + 2
    ids_list, masks_list = [], []
    for s in sentences:
        tok = tokenizer.encode(s, add_special_tokens=False)
        tok = tok[:max_sentence_tokens]
        seq = tok + [eos_id, srep_id]
        if len(seq) < L:
            pad_len = L - len(seq)
            ids = seq + [pad_id] * pad_len
            mask = [1] * len(seq) + [0] * pad_len
        else:
            ids = seq[:L]
            mask = [1] * L
        ids_list.append(torch.tensor(ids, dtype=torch.long))
        masks_list.append(torch.tensor(mask, dtype=torch.long))
    return ids_list, masks_list


def build_document_for_example(
    ex: Dict[str, Any],
    tokenizer: GPT2Tokenizer,
    splitter_args: Dict[str, Any],
    max_sentence_tokens: int,
) -> Dict[str, Any]:
    from sentence_splitter import create_token_based_sentence_splitter

    splitter = create_token_based_sentence_splitter(
        tokenizer=tokenizer,
        use_model=splitter_args["use_model_splitter"],
        model_name=splitter_args["splitter_model_name"],
        sentence_threshold=splitter_args["sentence_threshold"],
        max_sentence_tokens=max_sentence_tokens,
        min_sentence_tokens=splitter_args["min_sentence_tokens"],
    )
    sent_texts: List[str] = []
    for title, sents in ex.get("context", []):
        paragraph = " ".join(sents)
        sent_texts.extend(splitter.split_text(paragraph))
    ids_list, masks_list = pack_sentences_like_training(sent_texts, tokenizer, max_sentence_tokens)
    return {
        "sentences": ids_list,
        "attention_masks": masks_list,
        "sentence_texts": sent_texts,
        "doc_id": ex["_id"],
        "original_doc_id": ex["_id"],
        "num_sentences": len(sent_texts),
        "source_file": "hotpot",
    }


# --------------------------------------------
# Probe sentence construction for QA prompting
# --------------------------------------------

def make_probe_tokens(
    prefix_text: str,
    tokenizer: GPT2Tokenizer,
    max_sentence_tokens: int,
    reserve_tokens: int = 4,  # keep a little headroom if you ever extend the prefix
) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
    """Build a fixed-length probe sentence in training layout.

    IMPORTANT: We report the *true capacity* for candidate tokens as (L-2)-prefix_len,
    regardless of where [EOS][S_REP] appear in this seed probe. The replacement routine
    will re-synthesize the sentence tail to position [EOS][S_REP] immediately after the
    prefix + candidate tokens.
    """
    eos_id = tokenizer.convert_tokens_to_ids(SPECIALS["eos_token"])  # noqa: N806
    srep_id = tokenizer.convert_tokens_to_ids(SPECIALS["srep_token"])  # noqa: N806
    pad_id = tokenizer.pad_token_id
    L = max_sentence_tokens + 2

    prefix_ids_full = tokenizer.encode(prefix_text, add_special_tokens=False)
    max_prefix = max(0, max_sentence_tokens - int(reserve_tokens))
    trimmed = False
    if len(prefix_ids_full) > max_prefix:
        prefix_ids = prefix_ids_full[:max_prefix]
        trimmed = True
    else:
        prefix_ids = prefix_ids_full

    # Seed probe (will be reshaped by replacement):
    seq = prefix_ids + [eos_id, srep_id]
    if len(seq) < L:
        pad_len = L - len(seq)
        ids = seq + [pad_id] * pad_len
        mask = [1] * len(seq) + [0] * pad_len
    else:
        ids = seq[:L]
        mask = [1] * L

    # Debug: *theoretical* EOS/S_REP positions after replacement
    srep_pos = L - 1
    eos_pos = L - 2
    room = max(0, eos_pos - len(prefix_ids))
    dbg = {
        "L": L,
        "reserve": int(reserve_tokens),
        "prefix_token_count": len(prefix_ids_full),
        "prefix_len": len(prefix_ids),
        "prefix_trimmed": bool(trimmed),
        "eos_pos": int(eos_pos),  # theoretical positions
        "srep_pos": int(srep_pos),
        "room": int(room),  # true capacity for candidate tokens
    }
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        len(prefix_ids),
        dbg,
    )


def replace_probe_with_candidate(
    probe_ids: torch.Tensor,
    probe_mask: torch.Tensor,
    tokenizer: GPT2Tokenizer,
    candidate_text: str,
    prefix_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], Dict[str, Any]]:
    eos_id = tokenizer.convert_tokens_to_ids(SPECIALS["eos_token"])  # noqa: N806
    srep_id = tokenizer.convert_tokens_to_ids(SPECIALS["srep_token"])  # noqa: N806
    pad_id = tokenizer.pad_token_id

    ids = probe_ids.clone().tolist()
    L = len(ids)

    cand_ids_pre = tokenizer.encode(candidate_text, add_special_tokens=False)
    # Use the *full* capacity: positions [prefix_len, L-2) are available for candidate tokens
    max_cand = max(0, (L - 2) - prefix_len)
    cand_ids = cand_ids_pre[:max_cand]

    new_ids = ids[:prefix_len] + cand_ids + [eos_id, srep_id]
    if len(new_ids) < L:
        new_ids = new_ids + [pad_id] * (L - len(new_ids))
    else:
        new_ids = new_ids[:L]
    new_mask = [1 if t != pad_id else 0 for t in new_ids]

    # Debug from the updated layout
    srep_pos_new = new_ids.index(srep_id)
    eos_pos_new = srep_pos_new - 1
    dbg = {
        "cand_tokens_pre": len(cand_ids_pre),
        "cand_tokens_post": len(cand_ids),
        "start": int(prefix_len),
        "end": int(prefix_len + len(cand_ids)),
        "no_room": bool(len(cand_ids) == 0),
        "eos_pos": int(eos_pos_new),
        "srep_pos": int(srep_pos_new),
        "L": int(L),
    }
    return (
        torch.tensor(new_ids, dtype=torch.long),
        torch.tensor(new_mask, dtype=torch.long),
        cand_ids,
        dbg,
    )


def _inject_question_sentence(
    base_doc: Dict[str, Any],
    question: str,
    tokenizer: GPT2Tokenizer,
    max_sentence_tokens: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Encode 'Question: ...' as its own training-shaped sentence and append it so
    its [S_REP] goes into STM before the answer sentence.
    """
    q_txt = f"Question: {question}"
    q_ids_list, q_masks_list = pack_sentences_like_training([q_txt], tokenizer, max_sentence_tokens)
    q_ids, q_mask = q_ids_list[0].to(device), q_masks_list[0].to(device)
    return {
        **base_doc,
        "sentences": base_doc["sentences"] + [q_ids],
        "attention_masks": base_doc["attention_masks"] + [q_mask],
        "sentence_texts": base_doc["sentence_texts"] + [q_txt],
        "num_sentences": base_doc["num_sentences"] + 1,
    }


# --------------------------
# Scoring logic
# --------------------------

def _run_steps(model, ltm, documents, device) -> Dict[str, Any]:
    last = None
    for rec in model.iter_document_steps(
        [documents],
        ltm=ltm,
        warmup_weight=1.0,
        collect_debug=False,
        collect_ltm_debug=False,
        context_dropout_now=0.0,
    ):
        last = rec
    return last


def score_candidate(
    model,
    ltm,
    base_doc,
    probe_ids,
    probe_mask,
    prefix_len,
    cand_ids_list,
    device,
    tokenizer: Optional[GPT2Tokenizer] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (mean_log_prob, details_dict).
    If there is no room (len(cand_ids_list)==0), returns (-inf, {reason: "NO_ROOM", ...}).
    """
    idslist = probe_ids.tolist()
    L = len(idslist)

    if len(cand_ids_list) == 0:
        return float("-inf"), {
            "reason": "NO_ROOM",
            "prefix_len": int(prefix_len),
            "eos_pos": int(L - 2),
            "srep_pos": int(L - 1),
            "L": int(L),
        }

    doc = dict(base_doc)
    doc = {
        **doc,
        "sentences": doc["sentences"] + [probe_ids.to(device)],
        "attention_masks": doc["attention_masks"] + [probe_mask.to(device)],
        "sentence_texts": doc["sentence_texts"] + ["__PROBE__"],
        "num_sentences": doc["num_sentences"] + 1,
    }
    rec = _run_steps(model, ltm, doc, device)
    logits = rec["logits"]  # [1, L, V]
    ids = rec["input_ids"]  # [1, L]
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]

    valid = torch.zeros_like(shift_labels, dtype=torch.bool)
    start = int(prefix_len)
    end = int(start + len(cand_ids_list))  # score exactly the candidate span
    valid[0, start:end] = True

    labels = shift_labels.masked_fill(~valid, -100)
    logp = F.log_softmax(shift_logits, dim=-1)
    token_logp = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp[valid]
    mean_logp = float(token_logp.mean().item())

    probe_tail = None
    if tokenizer is not None:
        tail_ids = ids[0].tolist()
        probe_tail = tokenizer.decode(tail_ids, clean_up_tokenization_spaces=False)

    details = {
        "start": start,
        "end": end,
        "per_token_logp": [float(x) for x in token_logp.detach().cpu().tolist()],
        "prefix_len": int(prefix_len),
        "eos_pos": int(L - 2),
        "srep_pos": int(L - 1),
        "L": int(L),
        "probe_after_candidate_decoded": probe_tail,
    }
    return mean_logp, details


# --------------------------
# Span candidate generation
# --------------------------

def _simple_overlap_topk(question: str, sentences: List[str], k: int = 8) -> List[int]:
    import re as _re

    def toks(s):
        return [w for w in _re.findall(r"[a-z0-9]+", s.lower()) if w]

    q = set(toks(question))
    scored = []
    for i, s in enumerate(sentences):
        ss = set(toks(s))
        score = len(q & ss)
        scored.append((score, -i))
    scored.sort(reverse=True)
    keep = [(-idx) for score, idx in scored[: max(1, k)] if score > 0 or k >= len(sentences)]
    if not keep:
        keep = list(range(min(k, len(sentences))))
    return keep


def _enumerate_spans_from_sentence(text: str, max_len_words: int) -> List[str]:
    words = text.split()
    n = len(words)
    spans = []
    for i in range(n):
        for Lw in range(1, max_len_words + 1):
            j = i + Lw
            if j <= n:
                spans.append(" ".join(words[i:j]))
    return spans


def best_span_by_scoring(
    model,
    ltm,
    base_doc,
    question: str,
    tokenizer: GPT2Tokenizer,
    max_sentence_tokens: int,
    span_sent_topk: int,
    span_max_len: int,
    span_max_cands: int,
    probe_reserve: int,
    device,
) -> Tuple[str, float, Dict[str, Any]]:
    prefix = f"Question: {question} Answer: "
    probe_ids, probe_mask, prefix_len, probe_dbg = make_probe_tokens(
        prefix, tokenizer, max_sentence_tokens, reserve_tokens=probe_reserve
    )

    sent_idx = _simple_overlap_topk(question, base_doc["sentence_texts"], k=span_sent_topk)
    candidates = []
    for idx in sent_idx:
        s = base_doc["sentence_texts"][idx]
        spans = _enumerate_spans_from_sentence(s, max_len_words=span_max_len)
        for sp in spans:
            candidates.append((idx, sp))
            if len(candidates) >= span_max_cands:
                break
        if len(candidates) >= span_max_cands:
            break

    if not candidates:
        return "", float("-inf"), {"candidates": [], "scores_top": [], "probe": probe_dbg}

    best_span, best_score = "", float("-inf")
    scores = []
    for idx, span_text in candidates:
        ids_cand, mask_cand, cand_ids, repl_dbg = replace_probe_with_candidate(
            probe_ids, probe_mask, tokenizer, span_text, prefix_len
        )
        s, s_dbg = score_candidate(
            model, ltm, base_doc, ids_cand, mask_cand, prefix_len, cand_ids, device, tokenizer
        )
        scores.append((s, idx, span_text, {**repl_dbg, **s_dbg}))
        if s > best_score:
            best_score = s
            best_span = span_text

    scores_sorted = sorted(scores, key=lambda t: t[0], reverse=True)[:10]
    debug = {
        "probe": probe_dbg,
        "sent_topk_idx": sent_idx,
        "candidates_considered": len(candidates),
        "scores_top": [
            {"score_mean_logp": float(s), "sent_index": int(i), "text": t, "dbg": d}
            for (s, i, t, d) in scores_sorted
        ],
    }
    return best_span, best_score, debug


# -------------------------
# EM/F1 via official module
# -------------------------

def compute_em_f1(pred: str, gold: str, hp_mod) -> Tuple[float, float]:
    if hasattr(hp_mod, "exact_match_score"):
        em = float(hp_mod.exact_match_score(pred, gold))
    else:
        na = hp_mod.normalize_answer if hasattr(hp_mod, "normalize_answer") else (lambda s: s.strip().lower())
        em = float(na(pred) == na(gold))
    if hasattr(hp_mod, "f1_score"):
        f1, _prec, _rec = hp_mod.f1_score(pred, gold)
        return em, float(f1)
    # fallback
    na = hp_mod.normalize_answer if hasattr(hp_mod, "normalize_answer") else (lambda s: s.strip().lower())
    gt = na(gold).split()
    pd = na(pred).split()
    common: Dict[str, int] = {}
    for t in pd:
        common[t] = min(pd.count(t), gt.count(t))
    num_same = sum(common.values())
    if len(gt) == 0 or len(pd) == 0:
        return float(gt == pd), float(gt == pd)
    if num_same == 0:
        return em, 0.0
    precision = 1.0 * num_same / len(pd)
    recall = 1.0 * num_same / len(gt)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, float(f1)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--ckpt_name", default="best")
    ap.add_argument("--hotpot_json", required=True)
    ap.add_argument("--hotpot_repo", required=True)
    ap.add_argument("--ltm_scope", choices=["document", "global"], default="document")
    ap.add_argument("--mode", choices=["span", "score", "auto"], default="score")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Sampling & logging
    ap.add_argument("--sample_n", type=int, default=1000)
    ap.add_argument("--sample_seed", type=int, default=1337)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default=None)

    # Span controls
    ap.add_argument("--span_sent_topk", type=int, default=8)
    ap.add_argument("--span_max_len", type=int, default=8)
    ap.add_argument("--span_max_cands", type=int, default=1500)

    # Probe robustness
    ap.add_argument("--probe_reserve", type=int, default=4, help="Reserve at least this many tokens for the answer")

    # Optional restriction for score mode
    ap.add_argument(
        "--restrict_to_yesno",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1 and mode=score, skip non-yes/no items.",
    )

    # Optional overrides (defaults taken from training args)
    ap.add_argument("--override_use_ltm", type=int, choices=[0, 1], default=None)
    ap.add_argument("--override_debug_no_memory", type=int, choices=[0, 1], default=None)
    ap.add_argument("--override_debug_stm_only", type=int, choices=[0, 1], default=None)
    ap.add_argument("--override_ltm_query_mode", type=str, choices=["tokens_only", "hybrid", "both"], default=None)
    ap.add_argument("--override_ltm_top_k", type=int, default=None)
    ap.add_argument("--override_ltm_min_sim", type=float, default=None)
    ap.add_argument("--override_ltm_min_matches", type=int, default=None)
    ap.add_argument("--override_max_sentence_tokens", type=int, default=None)

    args = ap.parse_args()
    device = torch.device(args.device)

    # Load args/config
    train_args = _load_train_args(args.exp_dir, args.ckpt_name)
    tokenizer = _build_tokenizer(train_args)
    cfg = _cfg_from_args(train_args, tokenizer)
    cfg = _apply_overrides(cfg, args)

    # Model & weights
    model = SentenceTransformer(cfg, tokenizer).to(device)
    ckpt_dir = os.path.join(args.exp_dir, "checkpoints", args.ckpt_name)
    mdl_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.isfile(mdl_path):
        raise FileNotFoundError(f"Missing model state_dict at {mdl_path}")
    state = torch.load(mdl_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # LTM
    if cfg.use_long_term_memory and not cfg.debug_stm_only and not cfg.debug_no_memory:
        if args.ltm_scope == "document":
            ltm = DocumentLongTermMemory(cfg.srep_dim, device)
        else:
            ltm = LongTermMemoryGPU(cfg.srep_dim, device)
            ltm_path = os.path.join(ckpt_dir, "ltm.pt")
            if os.path.isfile(ltm_path):
                blob = torch.load(ltm_path, map_location="cpu")
                embs = blob.get("embeddings", None)
                texts = blob.get("texts", None)
                if embs is not None and texts:
                    ltm.embeddings = F.normalize(embs.to(device), p=2, dim=1)
                    ltm.texts = list(texts)
                    ltm.sentence_to_idx = {s: i for i, s in enumerate(ltm.texts)}
                    log.info(f"Seeded global LTM with {len(ltm.texts)} entries from training.")
    else:
        ltm = None

    # Data & official evaluator
    data_all = load_hotpot_json(args.hotpot_json)
    hp = _load_hotpot_module(args.hotpot_repo)

    # Sub-sample
    if args.sample_n and args.sample_n > 0 and args.sample_n < len(data_all):
        rng = random.Random(args.sample_seed)
        data = rng.sample(data_all, args.sample_n)
    else:
        data = data_all

    # Splitter args from train args
    splitter_args = dict(
        use_model_splitter=train_args.get("use_model_splitter", True),
        splitter_model_name=train_args.get("splitter_model_name", "sat-3l-sm"),
        sentence_threshold=train_args.get("sentence_threshold", 0.15),
        min_sentence_tokens=train_args.get("min_sentence_tokens", 3),
    )
    max_sentence_tokens = cfg.max_sentence_tokens

    # Prepare run folder
    if args.out_dir is None:
        import time

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(args.exp_dir, "eval_runs", f"{ts}_{args.mode}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    pred_path = os.path.join(out_dir, "predictions.json")
    summary_path = os.path.join(out_dir, "summary.json")

    def _open_log():
        return open(log_path, "a", encoding="utf-8")

    # Bookkeeping
    predictions: Dict[str, str] = {}
    em_sum, f1_sum, n_eval = 0.0, 0.0, 0
    n_skipped_non_yesno = 0
    decision_hist = {"span": 0, "score_yes": 0, "score_no": 0, "auto_span": 0, "auto_yes": 0, "auto_no": 0}

    pbar = tqdm(total=len(data), desc=f"Evaluating ({args.mode})")

    for idx_ex, ex in enumerate(data, start=1):
        qid = ex["_id"]
        question = ex["question"].strip()
        gold_ans = ex["answer"]
        gold_is_yesno = str(gold_ans).strip().lower() in ("yes", "no")

        if args.mode == "score" and args.restrict_to_yesno and not gold_is_yesno:
            n_skipped_non_yesno += 1
            pbar.update(1)
            continue

        doc = build_document_for_example(ex, tokenizer, splitter_args, max_sentence_tokens)
        if isinstance(ltm, DocumentLongTermMemory):
            ltm.reset_document(qid)

        chosen_text = ""
        details_for_log: Dict[str, Any] = {"qid": qid, "mode": args.mode}

        if args.mode == "score":
            # Put the question in STM as its own sentence, then score an ANSWER-only probe.
            doc_q = _inject_question_sentence(doc, question, tokenizer, max_sentence_tokens, device)
            prefix = "Answer: "
            probe_ids, probe_mask, prefix_len, probe_dbg = make_probe_tokens(
                prefix, tokenizer, max_sentence_tokens, reserve_tokens=args.probe_reserve
            )
            ids_yes, mask_yes, cand_yes, repl_yes = replace_probe_with_candidate(
                probe_ids, probe_mask, tokenizer, "yes", prefix_len
            )
            s_yes, yes_dbg = score_candidate(
                model, ltm, doc_q, ids_yes, mask_yes, prefix_len, cand_yes, device, tokenizer
            )

            ids_no, mask_no, cand_no, repl_no = replace_probe_with_candidate(
                probe_ids, probe_mask, tokenizer, "no", prefix_len
            )
            s_no, no_dbg = score_candidate(
                model, ltm, doc_q, ids_no, mask_no, prefix_len, cand_no, device, tokenizer
            )

            m = max(s_yes, s_no)
            if math.isfinite(m):
                p_yes = math.exp(s_yes - m) / (math.exp(s_yes - m) + math.exp(s_no - m))
                p_no = 1.0 - p_yes
            else:
                p_yes, p_no = float("nan"), float("nan")

            chosen_text = "yes" if s_yes >= s_no else "no"
            predictions[qid] = chosen_text
            decision_hist["score_yes" if chosen_text == "yes" else "score_no"] += 1

            details_for_log.update(
                {
                    "probe_summary": probe_dbg,
                    "score_yes_mean_logp": float(s_yes),
                    "score_no_mean_logp": float(s_no),
                    "prob_yes_softmax2": float(p_yes),
                    "prob_no_softmax2": float(p_no),
                    "yes_details": {**repl_yes, **yes_dbg},
                    "no_details": {**repl_no, **no_dbg},
                }
            )

        elif args.mode == "span":
            best_span, best_score, span_dbg = best_span_by_scoring(
                model,
                ltm,
                doc,
                question,
                tokenizer,
                max_sentence_tokens,
                args.span_sent_topk,
                args.span_max_len,
                args.span_max_cands,
                args.probe_reserve,
                device,
            )
            chosen_text = best_span or ""
            predictions[qid] = chosen_text
            decision_hist["span"] += 1
            details_for_log.update({"span_best_mean_logp": float(best_score), "span_debug": span_dbg})

        else:  # auto
            # Score branch (yes/no): inject question as its own sentence.
            doc_q = _inject_question_sentence(doc, question, tokenizer, max_sentence_tokens, device)
            prefix = "Answer: "
            probe_ids, probe_mask, prefix_len, probe_dbg = make_probe_tokens(
                prefix, tokenizer, max_sentence_tokens, reserve_tokens=args.probe_reserve
            )
            ids_yes, mask_yes, cand_yes, repl_yes = replace_probe_with_candidate(
                probe_ids, probe_mask, tokenizer, "yes", prefix_len
            )
            s_yes, yes_dbg = score_candidate(
                model, ltm, doc_q, ids_yes, mask_yes, prefix_len, cand_yes, device, tokenizer
            )

            ids_no, mask_no, cand_no, repl_no = replace_probe_with_candidate(
                probe_ids, probe_mask, tokenizer, "no", prefix_len
            )
            s_no, no_dbg = score_candidate(
                model, ltm, doc_q, ids_no, mask_no, prefix_len, cand_no, device, tokenizer
            )

            best_span, s_span, span_dbg = best_span_by_scoring(
                model,
                ltm,
                doc,
                question,
                tokenizer,
                max_sentence_tokens,
                args.span_sent_topk,
                args.span_max_len,
                args.span_max_cands,
                args.probe_reserve,
                device,
            )

            yn_best = max(s_yes, s_no)
            yn_label = "yes" if s_yes >= s_no else "no"

            if s_span >= yn_best:
                chosen_text = best_span or ""
                predictions[qid] = chosen_text
                decision_hist["auto_span"] += 1
                details_for_log.update(
                    {
                        "auto_choice": "span",
                        "probe_summary": probe_dbg,
                        "span_best_mean_logp": float(s_span),
                        "span_debug": span_dbg,
                        "score_yes_mean_logp": float(s_yes),
                        "score_no_mean_logp": float(s_no),
                    }
                )
            else:
                chosen_text = yn_label
                predictions[qid] = chosen_text
                decision_hist["auto_yes" if yn_label == "yes" else "auto_no"] += 1
                details_for_log.update(
                    {
                        "auto_choice": "score",
                        "probe_summary": probe_dbg,
                        "score_yes_mean_logp": float(s_yes),
                        "score_no_mean_logp": float(s_no),
                        "span_best_mean_logp": float(s_span),
                        "span_debug_topk": span_dbg.get("scores_top", []),
                    }
                )

        # Metrics (answer only)
        em, f1 = compute_em_f1(chosen_text, gold_ans, hp)
        em_sum += em
        f1_sum += f1
        n_eval += 1

        # Periodic log
        if (idx_ex % max(1, args.log_every) == 0) or (idx_ex == len(data)):
            with _open_log() as fh:
                fh.write("=" * 88 + "\n")
                fh.write(f"Item {idx_ex}/{len(data)} | qid={qid}\n")
                fh.write(f"Question: {question}\n")
                fh.write(f"Gold: {gold_ans}\n")
                fh.write(f"Pred: {chosen_text}\n")
                fh.write(f"EM={em:.0f}  F1={f1:.3f}  Mode={args.mode}\n")
                fh.write("\nContext (as sentences, with special tokens appended):\n")
                for si, s in enumerate(doc["sentence_texts"]):
                    fh.write(f"S{si:02d}: {s} {SPECIALS['eos_token']}{SPECIALS['srep_token']}\n")
                fh.write("\nDetails:\n")
                fh.write(json.dumps(details_for_log, ensure_ascii=False, indent=2) + "\n\n")

        pbar.update(1)

    pbar.close()

    # Save predictions
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump({"answer": predictions}, f, ensure_ascii=False)
    log.info(f"Saved predictions → {pred_path}")

    # Aggregate metrics
    metrics = {"em": (em_sum / max(1, n_eval)), "f1": (f1_sum / max(1, n_eval)), "n": n_eval}
    settings = {
        "mode": args.mode,
        "ltm_scope": args.ltm_scope,
        "sample_n": args.sample_n if args.sample_n else len(data),
        "restrict_to_yesno": bool(args.restrict_to_yesno),
        "probe_reserve": args.probe_reserve,
        "span_sent_topk": args.span_sent_topk,
        "span_max_len": args.span_max_len,
        "span_max_cands": args.span_max_cands,
        "decision_hist": decision_hist,
        "n_skipped_non_yesno": n_skipped_non_yesno,
        "overrides": {
            "use_ltm": cfg.use_long_term_memory,
            "debug_no_memory": cfg.debug_no_memory,
            "debug_stm_only": cfg.debug_stm_only,
            "ltm_query_mode": cfg.ltm_query_mode,
            "ltm_top_k": cfg.ltm_top_k,
            "ltm_min_sim": cfg.ltm_min_sim,
            "ltm_min_matches": cfg.ltm_min_matches,
            "max_sentence_tokens": cfg.max_sentence_tokens,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "settings": settings}, f, indent=2)
    log.info(f"Hotpot (answer-only) → EM={metrics['em']:.4f}  F1={metrics['f1']:.4f}")
    log.info(f"Saved summary → {summary_path}")
    log.info(f"Detailed log → {log_path}")


if __name__ == "__main__":
    main()
