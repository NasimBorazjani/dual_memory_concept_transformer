#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_binding.py

Behavioral evaluation for:
  (1) Object–feature attribution task (paper-style)
  (2) Father–son "reversal curse" task

Key properties:
- Reads training config from the run directory and replicates the same inference regime:
  * STM vs STM+LTM (and LTM scope), retrieval thresholds, max sentence tokens, etc.
- Supports two input modes:
  * Default: two-sentence protocol:
      - Sentence 1 (context) -> emits S_REP into STM (and LTM if enabled)
      - Sentence 2 (query prefix, trailing space) -> next-token logits at the last prefix token
  * --no_split mode:
      - Do NOT separate sentences; feed "sentence1 + query_prefix" as ONE segment
      - Score the next token right after the combined prefix (before final EOS/S_REP)
- Filters lexicons to single-token answers under your tokenizer (" word" encodes to exactly 1 token)
- Counterbalances conditions as in the paper; reports normal vs reversed order separately
- Metrics: top-1 global, candidate-max, top-k variants (global and candidate-restricted)
- TQDM progress bars, periodic console prints, and full sample logs (.txt + .jsonl) per task

Requirements:
- model.py from your training repo on PYTHONPATH
- torch, transformers, tqdm, numpy

"""

import os, re, json, sys, math, random, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# ---- import your model types ----
from model import (
    SentenceTransformerConfig,
    SentenceTransformer,
    LongTermMemoryGPU,
    DocumentLongTermMemory,
)

# --------------- utilities ---------------

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

def _decode_ids(tok: GPT2Tokenizer, ids: List[int]) -> List[str]:
    return [tok.decode([i], clean_up_tokenization_spaces=False) for i in ids]

def _ensure_trailing_space(prefix: str) -> str:
    return prefix if prefix.endswith(" ") else prefix + " "

# -- tokenizer helpers --

def build_tokenizer() -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    special = {"pad_token": "[PAD]", "eos_token": "[EOS]", "additional_special_tokens": ["[S_REP]"]}
    tok.add_special_tokens(special)
    return tok

def tok_id_for_answer(tok: GPT2Tokenizer, word: str) -> Optional[int]:
    """Return the single-token id for ' word' if it exists; else None."""
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

# -- model (two-sentence) helpers --

def _encode_sentence_row(tok: GPT2Tokenizer, text: str, L: int) -> Tuple[List[int], List[int], int]:
    """
    Returns (ids_row, mask_row, prefix_len_without_specials)
    Note: we don't truncate; we raise if >= L after appending EOS/S_REP.
    """
    ids = tok.encode(text, add_special_tokens=False)
    eos_id  = tok.convert_tokens_to_ids("[EOS]")
    srep_id = tok.convert_tokens_to_ids("[S_REP]")
    ids_e = ids + [eos_id, srep_id]
    if len(ids_e) > L:
        raise ValueError(f"Sentence too long for context L={L}: got {len(ids_e)} (incl. EOS,S_REP). Text='{text}'")
    pad_id = tok.pad_token_id
    ids_row  = ids_e + [pad_id] * (L - len(ids_e))
    mask_row = [1] * len(ids_e) + [0] * (L - len(ids_e))
    return ids_row, mask_row, len(ids)

@torch.no_grad()
def next_token_logits_two_sentence(
    model: SentenceTransformer,
    ltm,  # None, LongTermMemoryGPU, or DocumentLongTermMemory
    tok: GPT2Tokenizer,
    sentence1: str,
    query_prefix: str,
) -> torch.Tensor:
    """
    Build a one-document example with exactly two sentences and run through
    model.iter_document_steps_fixed(...). Return logits at the position right
    after the query prefix (B=1, V) as a CPU tensor.
    """
    device = next(model.parameters()).device
    L = model.cfg.max_position_embeddings

    ids1, m1, _           = _encode_sentence_row(tok, sentence1, L)
    ids2, m2, pref_len2   = _encode_sentence_row(tok, query_prefix, L)

    ex = {
        "sentences":       [torch.tensor(ids1, dtype=torch.long, device=device),
                            torch.tensor(ids2, dtype=torch.long, device=device)],
        "attention_masks": [torch.tensor(m1,   dtype=torch.long, device=device),
                            torch.tensor(m2,   dtype=torch.long, device=device)],
        "doc_id":          "eval_doc",
        "sentence_texts":  [sentence1, query_prefix],
        "original_doc_id": "eval_doc",
    }

    gen = model.iter_document_steps_fixed([ex], ltm=ltm, warmup_weight=1.0, collect_debug=False)
    _ = next(gen)   # step 1: context -> S_REP goes to STM
    step2 = next(gen)  # step 2: query prefix -> logits we need

    # logits shape: [B=1, L, V]; take the final visible position of the prefix
    return step2["logits"][0, pref_len2 - 1, :].detach().cpu()

# ---- single-pass (no-split) helpers ----

def _concat_as_one(sentence1: str, query_prefix: str) -> str:
    """
    Produce a single text segment that contains:
      '<sentence1> <query_prefix-with-trailing-space>'
    ensuring exactly one space between them and a trailing space at the end,
    so we can score the next token right after the combined prefix.
    """
    return sentence1.strip() + " " + _ensure_trailing_space(query_prefix)

@torch.no_grad()
def next_token_logits_single_segment(
    model: SentenceTransformer,
    ltm,  # None, LongTermMemoryGPU, or DocumentLongTermMemory
    tok: GPT2Tokenizer,
    sentence1: str,
    query_prefix: str,
) -> torch.Tensor:
    """
    Single-pass mode: feed context and query prefix as ONE sentence row.
    Return logits at the position right after the combined prefix.
    """
    device = next(model.parameters()).device
    L = model.cfg.max_position_embeddings

    combined = _concat_as_one(sentence1, query_prefix)
    ids, m, pref_len = _encode_sentence_row(tok, combined, L)
    ex = {
        "sentences":       [torch.tensor(ids, dtype=torch.long, device=device)],
        "attention_masks": [torch.tensor(m,   dtype=torch.long, device=device)],
        "doc_id":          "eval_doc",
        "sentence_texts":  [combined],
        "original_doc_id": "eval_doc",
    }
    step = next(model.iter_document_steps_fixed([ex], ltm=ltm, warmup_weight=1.0, collect_debug=False))
    return step["logits"][0, pref_len - 1, :].detach().cpu()

# --------------- run-dir loaders ---------------

def _try_parse_start_config_text(log_text: str) -> Optional[Dict[str, Any]]:
    """
    The trainer prints:
      "Starting training with configuration:\n{ ...JSON... }"
    We grab the first well-formed JSON block after that line.
    """
    marker = "Starting training with configuration:"
    i = log_text.find(marker)
    if i < 0:
        return None
    j = log_text.find("{", i)
    k = log_text.find("}\n", j)
    if j < 0 or k < 0:
        return None
    blob = log_text[j:k+1]
    try:
        return json.loads(blob)
    except Exception:
        # fallback: try to expand until the last '}' before the next blank line
        for end in range(k+1, min(len(log_text), j+10000)):
            if log_text[end] == "}":
                try:
                    return json.loads(log_text[j:end+1])
                except Exception:
                    pass
    return None

@dataclass
class DiscoveredConfig:
    args: Dict[str, Any]
    cfg: SentenceTransformerConfig
    tokenizer: GPT2Tokenizer
    ltm_scope: str  # "document" or "global"
    use_ltm: bool

import re, json
from pathlib import Path
import torch
from transformers import GPT2Tokenizer
from model import SentenceTransformer, SentenceTransformerConfig, LongTermMemoryGPU, DocumentLongTermMemory

def _read_run_config(run_dir: str) -> dict:
    """
    Parse the pretty-printed JSON block that train.py prints at start-up.
    Returns {} if not found; keys match the argparse names from training.
    """
    cfg = {}
    log_path = Path(run_dir) / "logs" / "experiment.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Starting training with configuration:\s*(\{.*?\})", text, re.S)
        if m:
            try:
                cfg = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    # Meta (optional)
    meta_path = Path(run_dir) / "checkpoints" / "best" / "meta.json"
    if meta_path.exists():
        try:
            cfg.setdefault("__meta__", json.loads(meta_path.read_text()))
        except Exception:
            pass
    return cfg


def _detect_ckpt_style(state: dict) -> dict:
    """
    Infer dims and projection style from the checkpoint.
    Returns:
      {
        'd_model': int,
        'srep_dim': int,
        'srep_head_depth': 1|>=2,
        'proj_style': 'in_proj'|'qkv'
      }
    """
    keys = list(state.keys())
    d_model = int(state["tok_emb.weight"].shape[1])

    has_qkv   = any(k.endswith("cross_attn.q_proj_weight") for k in keys)
    has_inproj = any(k.endswith("cross_attn.in_proj_weight") for k in keys)
    proj_style = "qkv" if has_qkv else "in_proj" if has_inproj else "in_proj"

    # S_REP head depth and output dim
    if "sentence_head.weight" in state:  # single Linear head
        srep_head_depth = 1
        srep_dim = int(state["sentence_head.weight"].shape[0])
    elif "sentence_head.3.weight" in state:  # MLP head, last Linear is index 3
        srep_head_depth = 2
        srep_dim = int(state["sentence_head.3.weight"].shape[0])
    else:
        # Fallback: if qkv present, infer from k_proj weight; else tie to d_model
        if has_qkv:
            kkey = next(k for k in keys if k.endswith("cross_attn.k_proj_weight"))
            srep_dim = int(state[kkey].shape[1])
            srep_head_depth = 2
        else:
            srep_dim = d_model
            srep_head_depth = 1

    return {
        "d_model": d_model,
        "srep_dim": srep_dim,
        "srep_head_depth": srep_head_depth,
        "proj_style": proj_style,
    }


def load_model_from_run_dir(run_dir: str):
    """
    Returns: model, disc: DiscoveredConfig, ltm_or_none
    """
    run = Path(run_dir)
    ckpt = run / "checkpoints" / "best" / "model.pt"
    if not ckpt.exists():
        ckpt = run / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Could not find a model checkpoint under {run_dir}")

    # Load weights on CPU first
    state = torch.load(ckpt, map_location="cpu")
    run_cfg = _read_run_config(run_dir)
    style   = _detect_ckpt_style(state)  # {'d_model', 'srep_dim', 'srep_head_depth', 'proj_style'}

    # Tokenizer with the same specials as training
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "eos_token": "[EOS]",
        "additional_special_tokens": ["[S_REP]"],
    })

    # Compose model config – prefer values from run_cfg, fallback to checkpoint-inferred/defaults
    cfg = SentenceTransformerConfig(
        d_model=int(run_cfg.get("d_model", style["d_model"])),
        n_heads=int(run_cfg.get("n_heads", 12)),
        n_layers=int(run_cfg.get("n_layers", 12)),
        ffw_mult=4,
        dropout=float(run_cfg.get("dropout", 0.1)),
        attn_dropout=float(run_cfg.get("attn_dropout", 0.1)),
        max_position_embeddings=int(run_cfg.get("context_length", 66)),
        vocab_size=len(tokenizer),

        # S_REP space and head
        srep_dim=int(run_cfg.get("srep_dim", style["srep_dim"])),
        srep_head_depth=int(run_cfg.get("srep_head_depth", style["srep_head_depth"])),
        srep_head_mult=int(run_cfg.get("srep_head_mult", 4)),
        srep_head_activation=str(run_cfg.get("srep_head_activation", "gelu")),

        # Memory knobs
        max_sentences_in_short_term=int(run_cfg.get("max_sentences_in_short_term", 15)),
        use_long_term_memory=bool(run_cfg.get("use_long_term_memory", False)),
        no_ltm_for_first_k_blocks=int(run_cfg.get("no_ltm_for_first_k_blocks", 4)),
        ltm_top_k=int(run_cfg.get("ltm_top_k", 15)),
        ltm_min_sim=float(run_cfg.get("ltm_min_sim", 0.1)),
        ltm_min_matches=int(run_cfg.get("ltm_min_matches", 2)),
        ltm_query_mode=str(run_cfg.get("ltm_query_mode", "tokens_only")),

        # STM positional (if any)
        use_stm_positional=bool(run_cfg.get("use_stm_positional", False)),
        stm_positional_weight=float(run_cfg.get("stm_positional_weight", 1.0)),

        # S_REP misc
        srep_dropout=float(run_cfg.get("srep_dropout", 0.1)),
        srep_norm_target=float(run_cfg.get("srep_norm_target", 1.0)),
        srep_norm_margin=float(run_cfg.get("srep_norm_margin", 0.1)),
        srep_norm_reg_weight=float(run_cfg.get("srep_norm_reg_weight", 0.001)),

        # Debug flags
        debug_no_memory=bool(run_cfg.get("debug_no_memory", False)),
        debug_stm_only=bool(run_cfg.get("debug_stm_only", False)),

        max_sentence_tokens=int(run_cfg.get("max_sentence_tokens", 64)),
    )

    # IMPORTANT: if the checkpoint used combined in-proj, make srep_dim == d_model
    # so your cross-attn modules materialize *.in_proj_weight rather than separate q/k/v.
    if style["proj_style"] == "in_proj":
        cfg.srep_dim = cfg.d_model

    # Build the exact model class used in training and load weights (tolerate older keys)
    model = SentenceTransformer(cfg, tokenizer)
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # LTM: honor run settings and debug_no_memory
    ltm_scope = str(run_cfg.get("ltm_scope", "document")).lower()
    if not cfg.use_long_term_memory or cfg.debug_no_memory:
        ltm = None
    else:
        ltm = DocumentLongTermMemory(cfg.srep_dim, device) if ltm_scope == "document" \
              else LongTermMemoryGPU(cfg.srep_dim, device)

    # Wrap everything into a proper dataclass so callers can do disc.cfg / disc.tokenizer, etc.
    disc = DiscoveredConfig(
        args=run_cfg,
        cfg=cfg,
        tokenizer=tokenizer,
        ltm_scope=ltm_scope,
        use_ltm=cfg.use_long_term_memory and not cfg.debug_no_memory,
    )

    return model, disc, ltm



# --------------- metrics helpers ---------------

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
      - target_prob (float), cand_probs (dict)
    """
    lp = logits.cpu().numpy()
    probs = _softmax_np(lp)
    vocab_rank = int((np.argsort(-probs) == target_id).nonzero()[0][0]) + 1  # 1-based
    top1_global = (vocab_rank == 1)
    topk_global = (vocab_rank <= max(1, opts.topk))

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

# --------------- object–feature task ---------------

@dataclass
class ObjFeatSample:
    obj1: str; color1: str; size1: str
    obj2: str; color2: str; size2: str
    query_feature: str  # "color" or "size"
    query_obj_idx: int  # 1 or 2
    order_color_first: bool  # True: "color and size", False: "size and color"
    reversed_object_order: bool  # evaluates positional sensitivity

def build_objfeat_sentence1(s: ObjFeatSample) -> str:
    # order of features inside each object's phrase
    if s.order_color_first:
        f1 = f"{s.color1} and {s.size1}"
        f2 = f"{s.color2} and {s.size2}"
    else:
        f1 = f"{s.size1} and {s.color1}"
        f2 = f"{s.size2} and {s.color2}"

    if not s.reversed_object_order:
        sent = f"The {s.obj1} is {f1} and the {s.obj2} is {f2}."
    else:
        # swap object order but keep the same mapping of feature phrases to their objects
        sent = f"The {s.obj2} is {f2} and the {s.obj1} is {f1}."
    return sent

def build_objfeat_query_prefix(s: ObjFeatSample) -> str:
    obj = s.obj1 if s.query_obj_idx == 1 else s.obj2
    return _ensure_trailing_space(f"So, the {s.query_feature} of the {obj} is")

def objfeat_target_and_candidates(tok: GPT2Tokenizer, s: ObjFeatSample) -> Tuple[int, List[int], Dict[str,int]]:
    target_word = (s.color1 if s.query_obj_idx == 1 else s.color2) if s.query_feature == "color" \
                  else (s.size1  if s.query_obj_idx == 1 else s.size2)
    target_id = tok_id_for_answer(tok, target_word)
    # four in-context features (paper metric)
    cand_words = [s.color1, s.size1, s.color2, s.size2]
    cand_ids = []
    for w in cand_words:
        tid = tok_id_for_answer(tok, w)
        if tid is None:
            # should not happen due to filtering; skip safely
            continue
        cand_ids.append(tid)
    return target_id, cand_ids, {w: tok_id_for_answer(tok, w) for w in cand_words}

def sample_objfeat_instances(
    rng: random.Random,
    objects: List[str], colors: List[str], sizes: List[str],
    n_per_condition: int
) -> List[ObjFeatSample]:
    """
    Builds a list containing both normal and reversed-object-order conditions,
    counterbalanced by feature order and query target (feature × object index).
    """
    out: List[ObjFeatSample] = []
    feats = ["color", "size"]
    obj_idx = [1, 2]
    orders = [True, False]  # color-first vs size-first

    # We aim for equal counts across (feature × object × order) within each condition.
    combos = [(f, oi, od) for f in feats for oi in obj_idx for od in orders]
    per_combo = max(1, n_per_condition // len(combos))

    def pick_two_distinct(xs: List[str]) -> Tuple[str,str]:
        a = rng.choice(xs)
        b = rng.choice([t for t in xs if t != a])
        return a, b

    # normal
    for (f, oi, od) in combos:
        for _ in range(per_combo):
            o1, o2 = pick_two_distinct(objects)
            c1, c2 = pick_two_distinct(colors)
            z1, z2 = pick_two_distinct(sizes)
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
            o1, o2 = pick_two_distinct(objects)
            c1, c2 = pick_two_distinct(colors)
            z1, z2 = pick_two_distinct(sizes)
            out.append(ObjFeatSample(
                obj1=o1, color1=c1, size1=z1,
                obj2=o2, color2=c2, size2=z2,
                query_feature=f, query_obj_idx=oi,
                order_color_first=od,
                reversed_object_order=True
            ))
    # If n_per_condition not divisible, we may be short; pad randomly
    while sum(1 for s in out if not s.reversed_object_order) < n_per_condition:
        out.append(rng.choice([x for x in out if not x.reversed_object_order]))
    while sum(1 for s in out if s.reversed_object_order) < n_per_condition:
        out.append(rng.choice([x for x in out if x.reversed_object_order]))
    rng.shuffle(out)
    return out

# --------------- father–son task ---------------

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

def sample_fatherson_instances(
    rng: random.Random, male_names: List[str], n_per_condition: int
) -> List[FatherSonSample]:
    out: List[FatherSonSample] = []
    def pick_two_distinct(xs: List[str]) -> Tuple[str,str]:
        a = rng.choice(xs)
        b = rng.choice([t for t in xs if t != a])
        return a, b
    # normal
    for _ in range(n_per_condition):
        f, s = pick_two_distinct(male_names)
        out.append(FatherSonSample(father=f, son=s, reversed_context=False))
    # reversed
    for _ in range(n_per_condition):
        f, s = pick_two_distinct(male_names)
        out.append(FatherSonSample(father=f, son=s, reversed_context=True))
    rng.shuffle(out)
    return out

# --------------- logging ---------------

def header_text_from_discovered(run_dir: str, disc: DiscoveredConfig, args: argparse.Namespace) -> str:
    cfg = disc.cfg
    lines = []
    lines.append("="*88)
    lines.append("MODEL CONFIG DISCOVERED FROM RUN DIRECTORY")
    lines.append("="*88)
    lines.append(f"run_dir: {run_dir}")
    lines.append(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    lines.append(f"n_layers={cfg.n_layers} d_model={cfg.d_model} n_heads={cfg.n_heads}")
    lines.append(f"max_position_embeddings={cfg.max_position_embeddings}  max_sentence_tokens={cfg.max_sentence_tokens}")
    lines.append(f"srep_dim={cfg.srep_dim}  use_long_term_memory={cfg.use_long_term_memory}  ltm_scope={disc.ltm_scope}")
    lines.append(f"no_ltm_for_first_k_blocks={cfg.no_ltm_for_first_k_blocks}  ltm_top_k={cfg.ltm_top_k}  "
                 f"ltm_min_sim={cfg.ltm_min_sim}  ltm_min_matches={cfg.ltm_min_matches}  ltm_query_mode={cfg.ltm_query_mode}")
    lines.append(f"use_stm_positional={cfg.use_stm_positional} stm_positional_weight={cfg.stm_positional_weight}")
    lines.append(f"debug_no_memory={cfg.debug_no_memory}  debug_stm_only={cfg.debug_stm_only}")
    lines.append("")
    lines.append("="*88)
    lines.append("EVALUATION SETTINGS")
    lines.append("="*88)
    lines.append(f"input_mode={'single-pass (no-split)' if getattr(args, 'no_split', False) else 'two-sentence'}")
    lines.append(f"n_per_condition={args.n_per_condition}  topk={args.topk}  candidate_topk={args.candidate_topk}")
    lines.append(f"print_every={args.print_every}")
    lines.append("")
    return "\n".join(lines)

# --------------- main evaluation ---------------

def run_object_feature_eval(
    model: SentenceTransformer, ltm, disc: DiscoveredConfig,
    objects: List[str], colors: List[str], sizes: List[str],
    out_dir: Path, n_per_condition: int, print_every: int,
    opts: MetricOptions, seed: int, no_split: bool
):
    tok = disc.tokenizer
    rng = random.Random(seed)

    # Prepare samples
    samples = sample_objfeat_instances(rng, objects, colors, sizes, n_per_condition)

    # Output files
    txt_path = out_dir / "object_feature_samples.txt"
    jsonl_path = out_dir / "object_feature_samples.jsonl"
    summary_path = out_dir / "object_feature_summary.json"

    with txt_path.open("w", encoding="utf-8") as ftxt, jsonl_path.open("w", encoding="utf-8") as fj:
        # progress & metrics
        total = len(samples)
        pbar = tqdm(total=total, desc="Object–Feature eval")
        agg = {
            "normal": {"top1_global":0, "cand_max":0, "topk_global":0, "cand_topk":0, "count":0,
                       "mean_global_rank":[], "mean_candidate_rank":[]},
            "reversed": {"top1_global":0, "cand_max":0, "topk_global":0, "cand_topk":0, "count":0,
                         "mean_global_rank":[], "mean_candidate_rank":[]},
        }

        for i, s in enumerate(samples, 1):
            sent1 = build_objfeat_sentence1(s)
            qpref = build_objfeat_query_prefix(s)
            logits = (next_token_logits_single_segment if no_split
                      else next_token_logits_two_sentence)(model, ltm, tok, sent1, qpref)

            target_id, cand_ids, cand_map = objfeat_target_and_candidates(tok, s)
            if target_id is None or any(cid is None for cid in cand_ids):
                # filtered single-token lists should prevent this
                pbar.update(1)
                continue

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

            # periodic console + txt logging
            if (i % max(1, print_every)) == 0:
                top_str = ", ".join([f"{w}:{p:.3f}" for _, w, p in stats["top_tokens"]])
                msg = (f"[ObjFeat {i}/{total}] cond={cond} | S1='{sent1}' | Q='{qpref}__?'\n"
                       f"  target={tok.decode([target_id], clean_up_tokenization_spaces=False).strip()} "
                       f"({stats['target_prob']:.3f}) | cand_max={stats['cand_max']} | "
                       f"top1_global={stats['top1_global']}\n"
                       f"  top-{max(5,opts.topk)}: {top_str}")
                tqdm.write(msg)
                ftxt.write(msg + "\n")

            # write jsonl
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
                "metrics": {k: (int(v) if isinstance(v, bool) else v) for k, v in stats.items() if k not in ("candidate_probs","top_tokens")},
                "candidate_probs": {tok.decode([i], clean_up_tokenization_spaces=False): p
                                    for i,p in stats["candidate_probs"].items()},
                "top_tokens": [{"id":i,"str":sstr,"prob":p} for (i,sstr,p) in stats["top_tokens"]],
            }
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pbar.update(1)
        pbar.close()

        # summary
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
    model: SentenceTransformer, ltm, disc: DiscoveredConfig,
    male_names: List[str],
    out_dir: Path, n_per_condition: int, print_every: int,
    opts: MetricOptions, seed: int, no_split: bool
):
    tok = disc.tokenizer
    rng = random.Random(seed + 13)

    samples = sample_fatherson_instances(rng, male_names, n_per_condition)

    txt_path = out_dir / "fatherson_samples.txt"
    jsonl_path = out_dir / "fatherson_samples.jsonl"
    summary_path = out_dir / "fatherson_summary.json"

    with txt_path.open("w", encoding="utf-8") as ftxt, jsonl_path.open("w", encoding="utf-8") as fj:
        total = len(samples)
        pbar = tqdm(total=total, desc="Father–Son eval")
        agg = {
            "normal": {"top1_global":0, "cand_max":0, "topk_global":0, "cand_topk":0, "count":0,
                       "mean_global_rank":[], "mean_candidate_rank":[]},
            "reversed": {"top1_global":0, "cand_max":0, "topk_global":0, "cand_topk":0, "count":0,
                         "mean_global_rank":[], "mean_candidate_rank":[]},
        }

        for i, s in enumerate(samples, 1):
            sent1 = build_fatherson_sentence1(s)
            qpref = build_fatherson_query_prefix(s)
            logits = (next_token_logits_single_segment if no_split
                      else next_token_logits_two_sentence)(model, ltm, tok, sent1, qpref)

            target_id, cand_ids, cand_map = fatherson_target_and_candidates(tok, s)
            if target_id is None or any(cid is None for cid in cand_ids):
                pbar.update(1); continue

            stats = compute_metrics_from_logits(logits, target_id, cand_ids, opts)
            stats["top_tokens"] = topk_tokens_from_logits(tok, logits, k=max(5, opts.topk))

            cond = "normal" if s.reversed_context else "reversed"
            agg[cond]["count"] += 1
            agg[cond]["top1_global"] += int(stats["top1_global"])
            agg[cond]["topk_global"] += int(stats["topk_global"])
            agg[cond]["cand_max"]    += int(stats["cand_max"])
            agg[cond]["cand_topk"]   += int(stats["cand_topk"])
            agg[cond]["mean_global_rank"].append(stats["global_rank"])
            agg[cond]["mean_candidate_rank"].append(stats["candidate_rank"])

            # periodic console + txt logging
            if (i % max(1, print_every)) == 0:
                top_str = ", ".join([f"{w}:{p:.3f}" for _, w, p in stats["top_tokens"]])
                msg = (f"[FatherSon {i}/{total}] cond={cond} | S1='{sent1}' | Q='{qpref}__?'\n"
                       f"  target={tok.decode([target_id], clean_up_tokenization_spaces=False).strip()} "
                       f"({stats['target_prob']:.3f}) | cand_max={stats['cand_max']} | "
                       f"top1_global={stats['top1_global']}\n"
                       f"  top-{max(5,opts.topk)}: {top_str}")
                tqdm.write(msg)
                ftxt.write(msg + "\n")

            # write jsonl
            rec = {
                "condition": cond,
                "sentence1": sent1,
                "query_prefix": qpref,
                "target_id": int(target_id),
                "target_str": tok.decode([target_id], clean_up_tokenization_spaces=False),
                "candidate_ids": [int(x) for x in cand_ids],
                "candidate_strs": _decode_ids(tok, cand_ids),
                "metrics": {k: (int(v) if isinstance(v, bool) else v) for k, v in stats.items() if k not in ("candidate_probs","top_tokens")},
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
                "acc_candidate_max": d["cand_max"]/n,    # for {F,S}, identical to “correct of two”
                "acc_candidate_topk": d["cand_topk"]/n,
                "mean_global_rank": float(np.mean(d["mean_global_rank"])) if d["mean_global_rank"] else None,
                "mean_candidate_rank": float(np.mean(d["mean_candidate_rank"])) if d["mean_candidate_rank"] else None,
            }
        with summary_path.open("w", encoding="utf-8") as fs:
            json.dump(summary, fs, indent=2)
    return str(summary_path)

# --------------- CLI ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Training run directory (produced by your trainer)")
    ap.add_argument("--lexicon_dir", type=str, required=True, help="Directory with colors.txt, sizes.txt, objects.txt, male_names.txt")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write logs and summaries")
    ap.add_argument("--n_per_condition", type=int, default=100, help="Number of prompts per condition (normal and reversed) for each task")
    ap.add_argument("--topk", type=int, default=5, help="Global top-k success (target in top-k of full vocab)")
    ap.add_argument("--candidate_topk", type=int, default=2, help="Candidate-restricted top-k success (e.g., 1=paper metric)")
    ap.add_argument("--print_every", type=int, default=10, help="Print a sample every N items")
    ap.add_argument("--no_split", action="store_true",
                    help="Do NOT split context and query; feed them as one segment and score at the end of the combined prefix.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # load model & config
    model, disc, ltm = load_model_from_run_dir(args.run_dir)
    header = header_text_from_discovered(args.run_dir, disc, args)
    with (out_dir / "config_detected.txt").open("w", encoding="utf-8") as f:
        f.write(header + "\n")

    # lexicons (+ single-token filtering under this tokenizer)
    tok = disc.tokenizer
    lex_dir = Path(args.lexicon_dir)
    objects = filter_single_token_words(tok, _read_text_file(lex_dir / "objects.txt"))
    colors  = filter_single_token_words(tok, _read_text_file(lex_dir / "colors.txt"))
    sizes   = filter_single_token_words(tok, _read_text_file(lex_dir / "sizes.txt"))
    male    = filter_single_token_words(tok, _read_text_file(lex_dir / "male_names.txt"))
    # female  = filter_single_token_words(tok, _read_text_file(lex_dir / "female_names.txt"))  # not needed for "son"

    # sanity checks
    if len(objects) < 2 or len(colors) < 2 or len(sizes) < 2:
        raise ValueError("Not enough single-token objects/colors/sizes after filtering.")
    if len(male) < 2:
        raise ValueError("Not enough single-token male names after filtering.")

    opts = MetricOptions(topk=max(1,args.topk), candidate_topk=max(1,args.candidate_topk))

    # write header also atop the sample logs
    for name in ["object_feature_samples.txt", "fatherson_samples.txt"]:
        with (out_dir / name).open("w", encoding="utf-8") as f:
            f.write(header + "\n")

    # run both tasks
    obj_summary_path = run_object_feature_eval(
        model, ltm, disc, objects, colors, sizes, out_dir,
        n_per_condition=args.n_per_condition, print_every=args.print_every, opts=opts, seed=args.seed, no_split=args.no_split
    )
    fs_summary_path = run_fatherson_eval(
        model, ltm, disc, male, out_dir,
        n_per_condition=args.n_per_condition, print_every=args.print_every, opts=opts, seed=args.seed, no_split=args.no_split
    )

    # short console recap
    print("\nDone. Summaries:")
    print(f"  Object–feature: {obj_summary_path}")
    print(f"  Father–son:     {fs_summary_path}")

if __name__ == "__main__":
    main()
