#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT‑2 baseline with:
 • cache-first loading (discovers external caches before rebuilding),
 • tqdm progress bars for data prep, train, and eval,
 • identical plotting front-end (your TrainingPlotter),
 • 'phase' series (always 'NoMem') and Loss Contributions row (100% Main),
 • GPT‑2 stock tokenizer for training (no 64-token sentence truncation),
 • continuous context chunking (default 1024) with --cross_document option.
"""

import os, sys, json, time, math, argparse, logging, hashlib, pickle, glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from tqdm.auto import tqdm

# ---------- Your pipeline utilities (filtering/splitting/analysis) ----------
from sentence_splitter import create_token_based_sentence_splitter
from create_datasets import (
    extract_all_documents_from_files,
    split_documents_by_length_3buckets,
    process_documents_for_training,
    write_analysis_files
)

def _norm_text(s: str) -> str:
    # casefold() is a bit stronger than lower() for Unicode;
    # split/join collapses internal whitespace.
    return " ".join((s or "").casefold().strip().split())

# ---------- Import YOUR TrainingPlotter to ensure identical plots ----------
try:
    from train import TrainingPlotter
except Exception:
    try:
        from training_plotter import TrainingPlotter
    except Exception as e:
        raise ImportError(
            "TrainingPlotter not found. Ensure your TrainingPlotter class "
            "is importable as `train.TrainingPlotter` or `training_plotter.TrainingPlotter`."
        ) from e


# ============================= Tokenizers =============================

def _build_processing_tokenizer_for_filtering() -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {
        "pad_token": "[PAD]",
        "eos_token": "[EOS]",
        "additional_special_tokens": ["[S_REP]"],
    }
    tok.add_special_tokens(special_tokens)
    return tok

def _build_stock_gpt2_tokenizer() -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token  # ensure padding id exists
    return tok


# ============================= Data helpers =============================

def _iter_sentences(processed_docs: List[Dict]) -> Tuple[List[str], List[str]]:
    sents, doc_ids = [], []
    for d in processed_docs:
        did = str(d.get("original_doc_id", d.get("doc_id", "")))
        for t in d.get("sentence_texts", []):
            sents.append(t); doc_ids.append(did)
    return sents, doc_ids

def _tokenize_sentences_full(sentences: List[str], tok: GPT2Tokenizer) -> List[List[int]]:
    out = []
    for s in tqdm(sentences, desc="Tokenizing sentences (GPT‑2)", unit="sent"):
        out.append(tok.encode(s, add_special_tokens=False))
    return out

def _build_chunks(
    sentence_token_ids: List[List[int]],
    context_length: int,
    cross_document: bool,
    sentence_doc_ids: Optional[List[str]] = None
) -> List[List[int]]:

    chunks: List[List[int]] = []
    buf_tokens: List[int] = []
    buf_doc: Optional[str] = None

    def flush():
        nonlocal buf_tokens
        if buf_tokens:
            chunks.append(buf_tokens)
            buf_tokens = []

    if cross_document:
        it = enumerate(sentence_token_ids)
    else:
        assert sentence_doc_ids is not None, "sentence_doc_ids required when cross_document=False"
        it = enumerate(zip(sentence_token_ids, sentence_doc_ids))
        buf_doc = sentence_doc_ids[0] if sentence_doc_ids else None

    pbar_total = sum(len(x) for x in sentence_token_ids)
    pbar = tqdm(total=pbar_total, desc="Packing chunks", unit="tok")

    if cross_document:
        for _, ids in it:
            k = 0
            while k < len(ids):
                space = context_length - len(buf_tokens)
                if space == 0:
                    flush()
                    space = context_length
                take = min(space, len(ids) - k)
                buf_tokens.extend(ids[k:k+take]); k += take; pbar.update(take)
        flush()
    else:
        for i, pair in it:
            ids, did = pair
            if buf_doc is None:
                buf_doc = did
            if did != buf_doc:
                flush()
                buf_doc = did
            k = 0
            while k < len(ids):
                space = context_length - len(buf_tokens)
                if space == 0:
                    flush(); space = context_length
                take = min(space, len(ids) - k)
                buf_tokens.extend(ids[k:k+take]); k += take; pbar.update(take)
        flush()

    pbar.close()
    return chunks

def _compute_repetition(processed_docs: List[Dict], tok: GPT2Tokenizer) -> Tuple[float, float, int]:
    from collections import defaultdict
    occ: Dict[Tuple[str,str], List[int]] = defaultdict(list)
    total_sent, total_tok = 0, 0
    for d in processed_docs:
        did = str(d.get("doc_id", ""))
        for t in d.get("sentence_texts", []):
            ids = tok.encode(t, add_special_tokens=False)
            total_sent += 1; total_tok += len(ids)
            occ[(did, _norm_text(t))].append(len(ids))
    rep_sent = sum((len(v)-1) for v in occ.values() if len(v) > 1)
    rep_tok  = sum(sum(v[1:]) for v in occ.values() if len(v) > 1)
    sr = rep_sent / total_sent if total_sent else 0.0
    tr = rep_tok  / total_tok  if total_tok  else 0.0
    return float(sr), float(tr), int(total_sent)

def _dataset_fingerprint(processed_docs: List[Dict]) -> str:
    h = hashlib.sha256()
    for d in processed_docs:
        did = str(d.get("doc_id", ""))
        for t in d.get("sentence_texts", []):
            h.update((did + "\n" + t + "\n").encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


# ============================= Cache I/O (cache-first) =============================

KNOWN_CACHE_PATTERNS = [
    "{split}_chunks.pkl",
    "{split}.chunks.pkl",
    "chunks_{split}.pkl",
    "{split}_chunks.npy",
    "{split}_chunks.npz",
    "{split}_chunks.pt",
    "{split}_chunks.jsonl",
]

def _load_chunks_from_file(path: str) -> Optional[List[List[int]]]:
    try:
        if path.endswith(".pkl"):
            obj = pickle.load(open(path, "rb"))
            if isinstance(obj, dict) and "chunks" in obj: obj = obj["chunks"]
            if isinstance(obj, list) and all(isinstance(x, (list, np.ndarray)) for x in obj):
                return [list(map(int, x)) for x in obj]
        elif path.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
            if arr.dtype == object:
                return [list(map(int, row)) for row in arr.tolist()]
            elif arr.ndim == 2:
                return [list(map(int, row)) for row in arr]
        elif path.endswith(".npz"):
            z = np.load(path, allow_pickle=True)
            if "chunks" in z.files:
                arr = z["chunks"]
            else:
                arr = z[z.files[0]]
            if arr.dtype == object:
                return [list(map(int, row)) for row in arr.tolist()]
            elif arr.ndim == 2:
                return [list(map(int, row)) for row in arr]
        elif path.endswith(".pt"):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict) and "chunks" in obj: obj = obj["chunks"]
            if isinstance(obj, list) and all(isinstance(x, (list, np.ndarray, torch.Tensor)) for x in obj):
                out = []
                for x in obj:
                    if isinstance(x, torch.Tensor): x = x.tolist()
                    out.append(list(map(int, x)))
                return out
        elif path.endswith(".jsonl"):
            out = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    j = json.loads(line)
                    if "input_ids" in j:
                        out.append([int(t) for t in j["input_ids"]])
            if out: return out
    except Exception:
        return None
    return None

def _discover_external_cache(cache_roots: List[str], split: str) -> Optional[str]:
    for root in cache_roots:
        if not root: continue
        if not os.path.isdir(root): continue
        # direct pattern hits
        for pat in KNOWN_CACHE_PATTERNS:
            for path in glob.glob(os.path.join(root, pat.format(split=split))):
                return path
        # generic scan
        hits = sorted(glob.glob(os.path.join(root, f"**/*{split}*chunk*.p*"), recursive=True))
        if hits:
            return hits[0]
    return None

def _save_chunks_pickle(path: str, chunks: List[List[int]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"chunks": chunks}, f)


# ============================= Grad helpers & eval =============================

def _count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(train)}

def _grad_l2_of(params) -> float:
    s = 0.0
    for p in params:
        if p is None: continue
        g = getattr(p, "grad", None)
        if g is None: continue
        if g.is_sparse: g = g.coalesce().values()
        s += float(g.detach().float().pow(2).sum().item())
    return math.sqrt(s) if s > 0.0 else 0.0

def collect_gpt2_component_grad_norms(model: GPT2LMHeadModel) -> Tuple[Dict[str,float], Dict[str,float], Dict[str,float]]:
    blocks = model.transformer.h
    per_block = []
    for blk in blocks:
        attn = [blk.attn.c_attn.weight, blk.attn.c_attn.bias, blk.attn.c_proj.weight, blk.attn.c_proj.bias]
        ffn  = [blk.mlp.c_fc.weight, blk.mlp.c_fc.bias, blk.mlp.c_proj.weight, blk.mlp.c_proj.bias]
        ln   = [blk.ln_1.weight, blk.ln_1.bias, blk.ln_2.weight, blk.ln_2.bias]
        norms = {
            "self_attn": _grad_l2_of(attn),
            "ffn": _grad_l2_of(ffn),
            "layernorm": _grad_l2_of(ln),
        }
        per_block.append(norms)
    if not per_block:
        z = {"self_attn":0.0, "ffn":0.0, "layernorm":0.0}
        return z, z, z
    n = len(per_block)
    avg  = {k: sum(b[k] for b in per_block)/float(n) for k in ("self_attn","ffn","layernorm")}
    blk0 = per_block[0]
    blkl = per_block[-1]
    return avg, blk0, blkl

@torch.no_grad()
def evaluate(model, dataloader, device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_tok, total_correct = 0.0, 0, 0
    bar = tqdm(dataloader, desc="Eval", unit="batch", leave=False)
    for batch in bar:
        ids    = batch["input_ids"].to(device)
        attn   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out    = model(input_ids=ids, attention_mask=attn, labels=labels)
        loss, logits = out.loss, out.logits

        shift_labels = labels[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        valid = shift_labels != -100
        preds = shift_logits.argmax(dim=-1)
        total_correct += (preds.eq(shift_labels) & valid).sum().item()
        tok = int(valid.sum().item()); total_tok += tok
        total_loss += float(loss.item()) * tok

        if total_tok > 0:
            bar.set_postfix(loss=total_loss/total_tok, acc=total_correct/total_tok)
    bar.close()
    avg = total_loss / max(1, total_tok)
    acc = total_correct / max(1, total_tok)
    model.train()
    return avg, acc

def _to_ppl(loss_value: float) -> float:
    try: return float(math.exp(min(80.0, float(loss_value))))
    except Exception: return float('inf')


# ============================= Main =============================

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_folder", type=str, required=True)
    ap.add_argument("--extensions", nargs="+", default=[".jsonl", ".txt", ".json", ".train", ".parquet", ".pq"])
    ap.add_argument("--sample_docs_per_split", type=int, default=10)
    ap.add_argument("--min_sentences_per_document", type=int, default=2)
    ap.add_argument("--max_chars_per_doc", type=int, default=-1)

    # Experiment
    ap.add_argument("--output_dir", type=str, default="experiments17_data_jsons_only")
    ap.add_argument("--exp_name", type=str, default="gpt2_baseline")

    # Filtering (selection only; GPT‑2 NOT truncated per sentence)
    ap.add_argument("--max_sentence_tokens", type=int, default=64)
    ap.add_argument("--min_sentence_tokens", type=int, default=3)
    ap.add_argument("--use_model_splitter", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)
    ap.add_argument("--splitter_model_name", type=str, default="sat-3l")
    ap.add_argument("--sentence_threshold", type=float, default=0.2)

    # Chunking
    ap.add_argument("--context_length", type=int, default=1024)
    ap.add_argument("--cross_document", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)

    # Cache
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="Optional explicit cache directory. Script will also probe <exp>/cache and <data_folder>/cache.")
    ap.add_argument("--read_only_cache", action="store_true",
                    help="If set, fail instead of rebuilding when cache not found.")
    ap.add_argument("--prefer_existing_cache", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)

    # Training
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--plot_every", type=int, default=1000)
    ap.add_argument("--epoch_window", type=int, default=10000)
    ap.add_argument("--comp_grad_every", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")

    args = ap.parse_args()

    # Logging
    logger = logging.getLogger("gpt2_baseline")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    # Experiment dirs
    exp_base = os.path.join(args.output_dir, args.exp_name)
    plots_dir = os.path.join(exp_base, "plots")
    logs_dir  = os.path.join(exp_base, "logs")
    ckpt_dir  = os.path.join(exp_base, "checkpoints")
    exp_cache = os.path.join(exp_base, "cache")
    for p in (plots_dir, logs_dir, ckpt_dir, exp_cache):
        os.makedirs(p, exist_ok=True)

    logger.info("Starting GPT‑2 baseline with configuration:\n%s", json.dumps(vars(args), indent=2))

    # Tokenizers
    stock_tok = _build_stock_gpt2_tokenizer()
    proc_tok  = _build_processing_tokenizer_for_filtering()

    # Splitter
    splitter = create_token_based_sentence_splitter(
        tokenizer=proc_tok,
        use_model=args.use_model_splitter,
        model_name=args.splitter_model_name,
        sentence_threshold=args.sentence_threshold,
        max_sentence_tokens=args.max_sentence_tokens,
        min_sentence_tokens=args.min_sentence_tokens,
    )

    # File gather
    def _collect_files(root: str, exts: List[str]) -> List[str]:
        got = []
        for r, _, fns in os.walk(root):
            for fn in fns:
                if any(fn.lower().endswith(ext) for ext in exts):
                    got.append(os.path.join(r, fn))
        got.sort()
        return got

    train_dir = os.path.join(args.data_folder, "train")
    val_dir   = os.path.join(args.data_folder, "val")
    explicit_splits = os.path.isdir(train_dir) and os.path.isdir(val_dir)

    # Ingest documents (with bars)
    if explicit_splits:
        logger.info("✓ Found explicit splits.")
        train_files = _collect_files(train_dir, args.extensions)
        val_files   = _collect_files(val_dir,   args.extensions)
        if not train_files or not val_files:
            raise ValueError("No data files found under explicit split directories.")

        logger.info("Extracting TRAIN documents...")
        train_docs = extract_all_documents_from_files(train_files, splitter, use_document_boundaries=True,
                                                     max_chars_per_doc=args.max_chars_per_doc)
        logger.info("Extracting VAL documents...")
        val_docs   = extract_all_documents_from_files(val_files, splitter, use_document_boundaries=True,
                                                     max_chars_per_doc=args.max_chars_per_doc)
    else:
        files = _collect_files(args.data_folder, args.extensions)
        if not files:
            raise ValueError("No data files found.")
        logger.info("Extracting documents...")
        docs = extract_all_documents_from_files(files, splitter, use_document_boundaries=True,
                                                max_chars_per_doc=args.max_chars_per_doc)
        logger.info("Splitting docs into train/val…")
        train_docs, val_docs = split_documents_by_length_3buckets(
            docs, tokenizer=proc_tok, splitter=splitter,
            min_sentences_per_document=args.min_sentences_per_document,
            train_val_split=0.9, seed=args.seed
        )

    # Process + chunk per split, but try to LOAD FROM CACHE FIRST
    def build_or_load_split(split_name: str, raw_docs: List[Dict]):
        # Where to look for caches
        cache_roots = [
            args.cache_dir,
            exp_cache,
            os.path.join(args.data_folder, "cache"),
            args.data_folder,
        ]

        # (A) try to load an external cache if requested
        if args.prefer_existing_cache:
            external = _discover_external_cache(cache_roots, split_name)
            if external:
                loaded = _load_chunks_from_file(external)
                if loaded:
                    logging.info(f"✓ Loaded {split_name} chunks from existing cache: {external} "
                                 f"(num_examples={len(loaded)})")
                    return loaded, {"built_fresh": False, "source": external}

        # (B) Otherwise, process & chunk (and create our own cache file keyed by processed fingerprint)
        logging.info(f"No external cache found for '{split_name}'. Processing and packing…")

        # Process documents for training (your filtering & sentence splitting)
        processed = process_documents_for_training(
            raw_docs, tokenizer=proc_tok, sentence_splitter=splitter,
            max_sentence_tokens=args.max_sentence_tokens,
            min_sentences_per_document=args.min_sentences_per_document,
            min_sentence_tokens_filter=args.min_sentence_tokens,
        )

        # Tokenize sentences with stock GPT‑2
        sentences, sent_doc_ids = _iter_sentences(processed)
        sent_ids = _tokenize_sentences_full(sentences, stock_tok)

        # Context chunk packing
        chunks = _build_chunks(
            sent_ids, context_length=args.context_length,
            cross_document=args.cross_document, sentence_doc_ids=sent_doc_ids
        )

        # Analysis + fingerprint (of processed set)
        rep_sent_ratio, rep_tok_ratio, total_sent_pre = _compute_repetition(processed, stock_tok)
        total_tokens_pre  = int(sum(len(x) for x in sent_ids))
        total_tokens_post = int(sum(len(c) for c in chunks))
        essentials = {
            "dataset": split_name,
            "totals": {
                "total_examples_postchunk": int(len(chunks)),
                "total_sentences_prechunk": int(total_sent_pre),
                "total_tokens_prechunk": int(total_tokens_pre),
                "total_tokens_postchunk": int(total_tokens_post),
            },
            "repetition": {
                "sentence_repeat_ratio_postchunk": float(rep_sent_ratio),
                "token_repeat_ratio_postchunk": float(rep_tok_ratio),
            },
        }
        proc_fp = _dataset_fingerprint(processed)
        cache_tag = f"{split_name}_gpt2ctx{args.context_length}_xd{int(args.cross_document)}_{proc_fp}.pkl"
        write_analysis_files(exp_cache, split_name, cache_tag, essentials)

        # Save our own cache for future runs
        save_path = os.path.join(exp_cache, cache_tag)
        _save_chunks_pickle(save_path, chunks)
        logging.info(f"✓ Saved {split_name} cache: {save_path} (num_examples={len(chunks)})")

        return chunks, {"built_fresh": True, "cache_path": save_path}

    train_chunks, train_info = build_or_load_split("train", train_docs)
    val_chunks,   val_info   = build_or_load_split("val",   val_docs)

    if args.read_only_cache and (train_info.get("built_fresh") or val_info.get("built_fresh")):
        raise RuntimeError("read_only_cache set but at least one split had to be rebuilt.")

    # DataLoader
    collator = DataCollatorForLanguageModeling(tokenizer=stock_tok, mlm=False)

    class _ChunkDS(Dataset):
        def __init__(self, chunks: List[List[int]]): self.chunks = chunks
        def __len__(self): return len(self.chunks)
        def __getitem__(self, i): return {"input_ids": self.chunks[i]}

    train_loader = DataLoader(_ChunkDS(train_chunks), batch_size=args.batch_size, shuffle=True,  num_workers=0, collate_fn=collator)
    val_loader   = DataLoader(_ChunkDS(val_chunks),   batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=0, collate_fn=collator)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = stock_tok.eos_token_id

    # Stats
    param_stats = _count_params(model)
    os.makedirs(logs_dir, exist_ok=True)
    json.dump(param_stats, open(os.path.join(logs_dir, "model_param_stats.json"), "w"), indent=2)
    logger.info("Model parameters: %s", param_stats)

    # Optimizer & scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    def lr_lambda(step):
        if step < args.warmup_steps: return float(step) / float(max(1, args.warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - args.warmup_steps)))
    sched  = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and args.fp16))

    # Plotter & metrics — use YOUR TrainingPlotter for IDENTICAL formatting
    plotter = TrainingPlotter(plots_dir, n_layers=model.config.n_layer)
    metrics: Dict[str, List] = {"n_layers": model.config.n_layer}
    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    # Helper to append "front end" series consistently
    def _append_train_frontend_series(step, loss_mean, acc_mean):
        metrics.setdefault("train_steps", []).append(step)
        metrics.setdefault("train_loss", []).append(loss_mean)
        metrics.setdefault("train_acc",  []).append(acc_mean)
        metrics.setdefault("train_ppl_main_token", []).append(_to_ppl(loss_mean))
        metrics.setdefault("lr", []).append(sched.get_last_lr()[0])
        if torch.cuda.is_available():
            metrics.setdefault("gpu_memory_gb", []).append(torch.cuda.max_memory_allocated()/1e9)
        elapsed = time.time() - t0
        metrics.setdefault("steps_per_second", []).append(step / max(1e-9, elapsed))
        # Keep the "Training Phases" panel visible (constant "NoMem")
        metrics.setdefault("phase", []).append("NoMem")
        # Keep the "Loss Component Contributions" panel (100% main)
        metrics.setdefault("main_loss_per_step", []).append(loss_mean)
        for k in ("bow_loss_per_step","adjacency_loss_per_step","alignment_loss_per_step","norm_reg_loss_per_step"):
            metrics.setdefault(k, []).append(0.0)

    # Train
    model.train()
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        running_loss_sum, running_tokens, running_correct = 0.0, 0, 0
        last_tick = time.time()

        bar = tqdm(train_loader, desc=f"Train (epoch {epoch+1}/{args.epochs})", unit="batch", leave=True, dynamic_ncols=True)
        for batch in bar:
            ids    = batch["input_ids"].to(device)
            attn   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda" and args.fp16)):
                out   = model(input_ids=ids, attention_mask=attn, labels=labels)
                loss  = out.loss
                logits= out.logits

            # token accuracy
            shift_labels = labels[:, 1:].contiguous()
            shift_logits = logits[:, :-1, :].contiguous()
            valid = shift_labels != -100
            preds = shift_logits.argmax(dim=-1)
            correct = (preds.eq(shift_labels) & valid).sum().item()
            tok     = int(valid.sum().item())
            running_correct += correct
            running_tokens  += tok
            running_loss_sum += float(loss.item()) * tok

            # backward
            scaled = scaler.scale(loss / max(1, args.gradient_accumulation_steps))
            scaled.backward()

            # component grad norms (avg + first/last) — BEFORE zeroing grads
            if args.comp_grad_every > 0 and (global_step % args.comp_grad_every == 0):
                avg, blk0, blkl = collect_gpt2_component_grad_norms(model)
                last_idx = model.config.n_layer - 1
                for k, v in avg.items():
                    metrics.setdefault(f"comp_grad_{k}_steps", []).append(global_step)
                    metrics.setdefault(f"comp_grad_{k}", []).append(float(v))
                for k, v in blk0.items():
                    metrics.setdefault(f"comp_grad_{k}_L0_steps", []).append(global_step)
                    metrics.setdefault(f"comp_grad_{k}_L0", []).append(float(v))
                for k, v in blkl.items():
                    metrics.setdefault(f"comp_grad_{k}_L{last_idx}_steps", []).append(global_step)
                    metrics.setdefault(f"comp_grad_{k}_L{last_idx}", []).append(float(v))

            if ((global_step + 1) % args.gradient_accumulation_steps) == 0:
                scaler.unscale_(optim)
                # record grad-norm (unclipped)
                total_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                metrics.setdefault("grad_norm_steps", []).append(global_step)
                metrics.setdefault("grad_norm", []).append(float(total_gnorm.item()))
                # clip & step
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            # periodic frontend logging + bars
            if global_step % 50 == 0:
                train_loss_mean = (running_loss_sum / running_tokens) if running_tokens > 0 else 0.0
                train_acc_mean  = (running_correct / running_tokens) if running_tokens > 0 else 0.0
                _append_train_frontend_series(global_step, train_loss_mean, train_acc_mean)

                # tqdm postfix to mimic your "frontend reporting"
                now = time.time()
                dt = max(1e-6, now - last_tick); last_tick = now
                sps = 50.0 / dt if dt > 0 else 0.0
                toks = tok  # last batch tokens; can also average
                postfix = dict(loss=f"{train_loss_mean:.4f}",
                               acc=f"{train_acc_mean:.4f}",
                               ppl=f"{_to_ppl(train_loss_mean):.2f}",
                               lr=f"{sched.get_last_lr()[0]:.2e}",
                               sps=f"{sps:.1f}")
                if torch.cuda.is_available():
                    postfix["memGB"] = f"{torch.cuda.max_memory_allocated()/1e9:.2f}"
                bar.set_postfix(postfix)

            # eval cadence (with tqdm bar)
            if args.eval_every > 0 and (global_step % args.eval_every == 0):
                val_loss, val_acc = evaluate(model, val_loader, device)
                metrics.setdefault("val_steps", []).append(global_step)
                metrics.setdefault("val_loss", []).append(val_loss)
                metrics.setdefault("val_acc",  []).append(val_acc)
                metrics.setdefault("val_ppl_main_token", []).append(_to_ppl(val_loss))
                logger.info(f"[Step {global_step}] Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}  "
                            f"Val PPL={_to_ppl(val_loss):.3f}")

                # best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))
                    json.dump({"global_step": global_step, "best_val_loss": best_val_loss},
                              open(os.path.join(ckpt_dir, "best_meta.json"), "w"), indent=2)

            # plots (identical front‑end via your plotter)
            if args.plot_every > 0 and (global_step % args.plot_every == 0):
                plotter.plot_all(metrics, global_step, epoch_window=args.epoch_window)

            global_step += 1

        bar.close()
        # End-of-epoch plots + snapshot
        plotter.plot_all(metrics, global_step, epoch_window=args.epoch_window)
        try: plotter.snapshot(epoch)
        except Exception: pass

    # Final eval + dump
    val_loss, val_acc = evaluate(model, val_loader, device)
    metrics.setdefault("val_steps", []).append(global_step)
    metrics.setdefault("val_loss", []).append(val_loss)
    metrics.setdefault("val_acc",  []).append(val_acc)
    metrics.setdefault("val_ppl_main_token", []).append(_to_ppl(val_loss))
    plotter.plot_all(metrics, global_step, epoch_window=args.epoch_window)

    json.dump({k: (list(v) if isinstance(v, list) else v) for k, v in metrics.items()},
              open(os.path.join(logs_dir, "metrics.json"), "w"), indent=2)

    mins = (time.time() - t0) / 60.0
    logger.info(f"Done in {mins:.1f} min. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
