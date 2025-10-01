#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference rollout mirroring training-time STM behaviour (LTM disabled).
- Standalone CLI: load experiment folder, rebuild tokenizer/config, run rollout.
- Optional prompt: seed STM directly from CLI text, skipping dataset sampling.
- Generation cap: up to --max_tokens (default 200) or --max_sentences (default 10).
- Sampling policy: greedy; seed from ground-truth first token when dataset context is used.
- STM flow: random document + start index (unless prompt-only), push S_REPs after each sentence.
- Debug file: <exp>/debug/val_inference_latest.txt with concise STM size tracking.
"""

import argparse
import copy
import glob
import inspect
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

from create_datasets import (
    extract_all_documents_from_files,
    process_documents_for_training,
    split_documents_by_length_3buckets,
)
from model import SentenceTransformer, SentenceTransformerConfig
from sentence_splitter import create_token_based_sentence_splitter

logger = logging.getLogger("inference_rollout")
logger.setLevel(logging.INFO)

SPECIALS = {"pad_token": "[PAD]", "eos_token": "[EOS]", "srep_token": "[S_REP]"}


def _compat_call(fn, pos_args=(), kw=None):
    kw = kw or {}
    sig = inspect.signature(fn)
    allowed = {k: v for k, v in kw.items() if k in sig.parameters}
    try:
        return fn(*pos_args, **allowed)
    except TypeError:
        pass
    try:
        return fn(*(pos_args[:1]), **allowed)
    except TypeError:
        pass
    try:
        return fn(*(pos_args[:1]))
    except TypeError:
        pass
    return fn(*pos_args)


def _first_present(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _load_train_args(exp_dir: str) -> Dict[str, Any]:
    meta = os.path.join(exp_dir, "checkpoints", "meta.json")
    if os.path.exists(meta):
        with open(meta, "r") as f:
            blob = json.load(f)
        if isinstance(blob, dict) and "args" in blob:
            return blob["args"]

    log_path = os.path.join(exp_dir, "logs", "experiment.log")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            txt = f.read()
        blocks = re.findall(r"\{[\s\S]*?\}", txt)
        for blob in reversed(blocks):
            try:
                args = json.loads(blob)
                if isinstance(args, dict) and (
                    "data_folder" in args or "output_dir" in args
                ):
                    return args
            except Exception:
                continue
    raise FileNotFoundError(
        "Could not find train args in checkpoints/meta.json or logs/experiment.log"
    )


def _build_tokenizer(args: Dict[str, Any]) -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens(
        {
            "pad_token": SPECIALS["pad_token"],
            "eos_token": SPECIALS["eos_token"],
            "additional_special_tokens": [SPECIALS["srep_token"]],
        }
    )
    return tok


def _cfg_from_args(args: Dict[str, Any], tokenizer: GPT2Tokenizer) -> SentenceTransformerConfig:
    return SentenceTransformerConfig(
        d_model=int(args.get("d_model", 768)),
        n_heads=int(args.get("n_heads", 12)),
        n_layers=int(args.get("n_layers", 12)),
        ffw_mult=int(args.get("ffw_mult", 4)),
        dropout=float(args.get("dropout", 0.1)),
        attn_dropout=float(args.get("attn_dropout", 0.1)),
        srep_dim=int(args.get("srep_dim", 1024)),
        max_position_embeddings=int(args.get("max_position_embeddings", 66)),
        vocab_size=len(tokenizer),
        max_sentence_tokens=int(args.get("max_sentence_tokens", 64)),
        max_sentences_in_short_term=int(args.get("max_sentences_in_short_term", 15)),
        use_long_term_memory=bool(args.get("use_long_term_memory", False)),
        no_ltm_for_first_k_blocks=int(args.get("no_ltm_for_first_k_blocks", 4)),
        ltm_top_k=int(args.get("ltm_top_k", 5)),
        ltm_min_sim=float(args.get("ltm_min_sim", 0.3)),
        ltm_min_matches=int(args.get("ltm_min_matches", 2)),
        ltm_query_mode=str(args.get("ltm_query_mode", "hybrid")),
        use_stm_positional=bool(args.get("use_stm_positional", True)),
        stm_positional_weight=float(args.get("stm_positional_weight", 1.0)),
        memory_gate_init=float(args.get("memory_gate_init", 1.0)),
        context_dropout=float(args.get("context_dropout", 0.1)),
        srep_dropout=float(args.get("srep_dropout", 0.1)),
        srep_norm_target=float(args.get("srep_norm_target", 1.0)),
        srep_norm_margin=float(args.get("srep_norm_margin", 0.1)),
        use_attentive_pool=bool(args.get("use_attentive_pool", False)),
        debug_no_memory=bool(args.get("debug_no_memory", False)),
        debug_stm_only=bool(args.get("debug_stm_only", False)),
    )


def collate_documents(batch):
    return batch


def _collect_files(root: str, extensions: List[str]) -> List[str]:
    files: List[str] = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            if any(fn.lower().endswith(ext.lower()) for ext in extensions):
                files.append(os.path.join(r, fn))
    return sorted(files)


def _prep_loaders(
    args: Dict[str, Any], tokenizer: GPT2Tokenizer, splitter
) -> DataLoader:
    data_root = _first_present(
        args,
        ["data_folder", "data_root", "dataset_dir", "input_folder", "train_data_dir"],
    )
    if not data_root:
        raise KeyError(
            "Expected one of data_folder/data_root/dataset_dir/input_folder/train_data_dir in saved args"
        )

    exts = args.get("extensions", [".jsonl", ".txt", ".json", ".train", ".parquet", ".pq"])
    max_chars = int(args.get("max_chars_per_doc", -1))
    min_sents = int(args.get("min_sentences_per_article", 2))

    extractor_kwargs = {
        "extensions": exts,
        "max_chars_per_doc": max_chars,
        "min_sentences_per_article": min_sents,
    }

    val_docs: List[Dict[str, Any]]
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        val_files = _collect_files(val_dir, exts)
        if not val_files:
            raise ValueError(f"No validation files found under {val_dir}")
        val_docs = _compat_call(
            extract_all_documents_from_files,
            (val_files, splitter),
            extractor_kwargs,
        )
    else:
        all_files = _collect_files(data_root, exts)
        if not all_files:
            raise ValueError(f"No data files found under {data_root}")
        all_docs = _compat_call(
            extract_all_documents_from_files,
            (all_files, splitter),
            extractor_kwargs,
        )
        split_kwargs = {
            "min_sentences_per_document": int(args.get("min_sentences_per_article", 2)),
            "train_val_split": float(
                _first_present(args, ["train_val_split", "train_val_ratio", "split_ratio"]) or 0.9
            ),
            "seed": int(args.get("seed", 42)),
        }
        split_result = _compat_call(
            split_documents_by_length_3buckets,
            (all_docs, tokenizer, splitter),
            split_kwargs,
        )
        if isinstance(split_result, dict):
            val_docs = (
                split_result.get("val")
                or split_result.get("test")
                or []
            )
        else:
            val_docs = split_result[1] if len(split_result) > 1 else []
        if not val_docs:
            raise ValueError("Dataset split produced no validation documents")

    proc_kwargs = {
        "max_sentence_tokens": int(args.get("max_sentence_tokens", 64)),
        "min_sentence_tokens_filter": int(args.get("min_sentence_tokens_filter", 4)),
        "max_sentences_per_document": int(args.get("max_sentences_per_document", 32)),
    }
    processed_val = _compat_call(
        process_documents_for_training,
        (val_docs, tokenizer, splitter),
        proc_kwargs,
    )

    return DataLoader(processed_val, batch_size=1, shuffle=True, collate_fn=collate_documents)


def _select_random_document(loader):
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and len(dataset) > 0:
        idx = random.randrange(len(dataset))
        return idx, dataset[idx]

    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:  # pragma: no cover
        raise RuntimeError("Loader is empty; cannot run rollout") from exc

    if isinstance(batch, list) and len(batch) > 0:
        return None, random.choice(batch)
    return None, batch


def _build_prompt_context(
      prompt: str,
      tokenizer: GPT2Tokenizer,
      cfg: SentenceTransformerConfig,
  ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
      """
      Convert CLI prompt text into STM-ready tensors.
      We avoid the SaT splitter here because short prompts often degrade into word-level fragments.
      """
      prompt = prompt.strip()
      if not prompt:
          raise ValueError("Prompt cannot be empty or whitespace only")

      # Split on sentence-ending punctuation; keep the punctuation on each sentence.
      raw_chunks = re.split(r"(?<=[.!?])\s+", prompt)
      sentences = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

      if not sentences:
          sentences = [prompt]

      # Merge any trailing fragment that ends with a comma/semicolon/etc back into the previous sentence.
      if len(sentences) >= 2 and sentences[-1] and sentences[-1][-1] in ",;:-":
          sentences[-2] = sentences[-2] + " " + sentences[-1]
          sentences = sentences[:-1]

      capacity = int(cfg.max_sentences_in_short_term)
      sentences = sentences[:capacity]

      L = cfg.max_position_embeddings
      pad_id = tokenizer.convert_tokens_to_ids(SPECIALS["pad_token"])
      eos_id = tokenizer.convert_tokens_to_ids(SPECIALS["eos_token"])
      srep_id = tokenizer.convert_tokens_to_ids(SPECIALS["srep_token"])

      ctx_ids: List[torch.Tensor] = []
      ctx_masks: List[torch.Tensor] = []
      ctx_texts: List[str] = []

      for sent in sentences:
          token_ids = tokenizer.encode(sent, add_special_tokens=False)
          token_ids = token_ids[: cfg.max_sentence_tokens] + [eos_id, srep_id]
          pad_len = max(0, L - len(token_ids))
          padded = token_ids + [pad_id] * pad_len
          mask = [1] * min(len(token_ids), L) + [0] * pad_len

          ctx_ids.append(torch.tensor(padded[:L], dtype=torch.long))
          ctx_masks.append(torch.tensor(mask[:L], dtype=torch.long))
          ctx_texts.append(sent)

      return ctx_ids, ctx_masks, ctx_texts

@torch.no_grad()
def rollout_from_loader(
    model: SentenceTransformer,
    tokenizer: GPT2Tokenizer,
    loader: Optional[DataLoader],
    out_path: str,
    max_tokens: int = 200,
    max_sentences: int = 10,
    seed: int = 42,
    prompt_context: Optional[Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]] = None,
):
    random.seed(seed)
    torch.manual_seed(seed)

    pad_id = tokenizer.convert_tokens_to_ids(SPECIALS["pad_token"])
    eos_id = tokenizer.convert_tokens_to_ids(SPECIALS["eos_token"])
    srep_id = tokenizer.convert_tokens_to_ids(SPECIALS["srep_token"])
    L = model.cfg.max_position_embeddings
    capacity = int(model.cfg.max_sentences_in_short_term)

    context_sentences: List[torch.Tensor] = []
    context_masks: List[torch.Tensor] = []
    context_texts: List[str] = []

    if prompt_context:
        context_sentences, context_masks, context_texts = [
            list(x) for x in prompt_context
        ]

    sample_doc: Dict[str, Any]
    doc_idx: Optional[int]

    if loader is not None:
        doc_idx, sample_doc = _select_random_document(loader)
        template = {
            k: copy.deepcopy(v)
            for k, v in sample_doc.items()
            if k not in {"sentences", "attention_masks", "sentence_texts", "num_sentences"}
        }
        total_sentences = int(
            sample_doc.get("num_sentences", len(sample_doc.get("sentences", [])))
        )
        start_cap = min(total_sentences, capacity - len(context_sentences))
        t_start = random.randint(0, start_cap) if total_sentences > 0 else 0

        for i in range(t_start):
            context_sentences.append(sample_doc["sentences"][i].detach().clone())
            context_masks.append(sample_doc["attention_masks"][i].detach().clone())
            context_texts.append(sample_doc.get("sentence_texts", [""])[i])
    else:
        doc_idx = None
        template = {"doc_id": "prompt_only"}
        total_sentences = 0
        t_start = 0

    context_doc = {
        "sentences": context_sentences,
        "attention_masks": context_masks,
        "sentence_texts": context_texts,
        "doc_id": template.get("doc_id", "unknown"),
    }

    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("INFERENCE ROLLOUT (STM only; LTM disabled)")
    lines.append(f"Doc ID: {context_doc.get('doc_id', 'unknown')}")
    if doc_idx is not None:
        lines.append(f"Validation dataset index: {doc_idx}")
    lines.append(f"Prompt sentences primed: {len(prompt_context[0]) if prompt_context else 0}")
    lines.append(f"Additional STM priming sentences from dataset: {t_start}")
    lines.append(f"Generation limits -> max tokens: {max_tokens}, max sentences: {max_sentences}")
    lines.append("=" * 90)
    lines.append("")

    if context_texts:
        lines.append("Context seen by the model before generation:")
        for i, txt in enumerate(context_texts):
            lines.append(f"  [{i:02d}] {txt}")
    else:
        lines.append("Context seen by the model before generation: <empty>")
    lines.append("")

    stm_size = 0
    if context_sentences:
        priming_records = list(
            model.iter_document_steps(
                [context_doc],
                ltm=None,
                warmup_weight=1.0,
                collect_debug=True,
                debug_max_samples=1,
                collect_stm_grad_refs=False,
                stm_grad_limit=0,
                collect_comp_grad_refs=False,
                comp_grad_limit=0,
                context_dropout_now=0.0,
            )
        )
        if priming_records:
            last = priming_records[-1]
            before = last.get("actual_stm_sizes", [0])[0]
            added = 1 if last.get("srep_embs") is not None and last["srep_embs"].numel() > 0 else 0
            stm_size = min(capacity, before + added)

    lines.append(f"STM size after priming: {stm_size} / {capacity}")
    lines.append("")

    total_tokens = 0
    generated = 0
    cur_index = t_start
    default_seed = tokenizer.encode(".", add_special_tokens=False)[:1] or [eos_id]

    def _decode(ids_list):
        toks = [tid for tid in ids_list if tid not in (pad_id, srep_id)]
        if toks and toks[-1] == eos_id:
            toks = toks[:-1]
        return tokenizer.decode(toks, clean_up_tokenization_spaces=False).strip()

    while generated < max_sentences and total_tokens < max_tokens:
        lines.append("-" * 70)
        lines.append(f"Generated sentence {generated + 1}")
        lines.append(f"STM size entering step: {stm_size} / {capacity}")

        if loader is not None and cur_index < total_sentences:
            gt_tensor = sample_doc["sentences"][cur_index]
            seed_token = int(gt_tensor[0].item())
            ground_truth = _decode(gt_tensor.tolist())
        else:
            seed_token = default_seed[0]
            ground_truth = None

        cur_ids = [seed_token]
        finished = False
        steps_this_sentence = 0

        while (
            not finished
            and total_tokens < max_tokens
            and steps_this_sentence < model.cfg.max_sentence_tokens
        ):
            ids = cur_ids + [eos_id, srep_id]
            padded = ids + [pad_id] * max(0, L - len(ids))
            mask_len = min(len(ids), L)
            mask = [1] * mask_len + [0] * (L - mask_len)

            pseudo_doc = {
                "sentences": list(context_sentences)
                + [torch.tensor(padded[:L], dtype=torch.long)],
                "attention_masks": list(context_masks)
                + [torch.tensor(mask[:L], dtype=torch.long)],
                "sentence_texts": list(context_texts) + [_decode(padded)],
                "doc_id": context_doc.get("doc_id", "unknown"),
            }
            step_records = list(
                model.iter_document_steps(
                    [pseudo_doc],
                    ltm=None,
                    warmup_weight=1.0,
                    collect_debug=False,
                    debug_max_samples=0,
                    collect_stm_grad_refs=False,
                    stm_grad_limit=0,
                    collect_comp_grad_refs=False,
                    comp_grad_limit=0,
                    context_dropout_now=0.0,
                )
            )
            if not step_records:
                break

            logits = step_records[-1]["logits"]
            pos = max(0, len(cur_ids) - 1)
            next_id = int(torch.argmax(logits[0, pos, :]).item())

            cur_ids.append(next_id)
            steps_this_sentence += 1
            total_tokens += 1
            if next_id == eos_id:
                finished = True

        if cur_ids and cur_ids[-1] == eos_id:
            cur_ids = cur_ids[:-1]

        final_ids = cur_ids[: model.cfg.max_sentence_tokens] + [eos_id, srep_id]
        pad_needed = max(0, L - len(final_ids))
        final_ids = final_ids + [pad_id] * pad_needed
        mask_len = min(len(cur_ids) + 2, L)
        final_mask = [1] * mask_len + [0] * (L - mask_len)

        final_tensor = torch.tensor(final_ids[:L], dtype=torch.long)
        final_mask_tensor = torch.tensor(final_mask[:L], dtype=torch.long)
        generated_text = _decode(final_ids)

        pseudo_doc_final = {
            "sentences": list(context_sentences) + [final_tensor],
            "attention_masks": list(context_masks) + [final_mask_tensor],
            "sentence_texts": list(context_texts) + [generated_text],
            "doc_id": context_doc.get("doc_id", "unknown"),
        }
        final_records = list(
            model.iter_document_steps(
                [pseudo_doc_final],
                ltm=None,
                warmup_weight=1.0,
                collect_debug=True,
                debug_max_samples=1,
                collect_stm_grad_refs=False,
                stm_grad_limit=0,
                collect_comp_grad_refs=False,
                comp_grad_limit=0,
                context_dropout_now=0.0,
            )
        )
        if final_records:
            step_info = final_records[-1]
            before_push = step_info.get("actual_stm_sizes", [stm_size])[0]
            added = 1 if step_info.get("srep_embs") is not None and step_info["srep_embs"].numel() > 0 else 0
            after_push = min(capacity, before_push + added)
        else:
            after_push = stm_size

        context_sentences.append(final_tensor)
        context_masks.append(final_mask_tensor)
        context_texts.append(generated_text)

        lines.append(f"STM size after push : {after_push} / {capacity}")
        if ground_truth is not None:
            lines.append(f"Ground-truth comparison: \"{ground_truth}\"")
        lines.append(f"Model output: \"{generated_text}\"")
        lines.append(f"Tokens generated so far: {total_tokens}/{max_tokens}")
        lines.append("")

        stm_size = after_push
        generated += 1
        cur_index += 1

        if total_tokens >= max_tokens:
            lines.append("Reached token budget; stopping rollout.")
            lines.append("")
            break

    text = "\n".join(lines)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--max_sentences", type=int, default=10)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--save_to_debug", type=bool, default=False,
                     help="If set, also write the rollout to <exp>/debug/val_inference_latest.txt")
    ap.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt text to seed STM; skips dataset sampling when provided.",
    )
    args = ap.parse_args()

    train_args = _load_train_args(args.exp_dir)
    tokenizer = _build_tokenizer(train_args)
    cfg = _cfg_from_args(train_args, tokenizer)

    model = SentenceTransformer(cfg, tokenizer)
    ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
    ckpt_files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith((".pt", ".bin"))
    ] if os.path.isdir(ckpt_dir) else []
    if ckpt_files:
        best = sorted(ckpt_files)[-1]
        state = torch.load(best, map_location=args.device)
        model.load_state_dict(state, strict=False)
    model = model.to(args.device)
    model.eval()

    splitter_kwargs = dict(
        tokenizer=tokenizer,
        use_model=_first_present(train_args, ["use_model_splitter", "splitter_use_model", "use_splitter_model"]) or False,
        model_name=_first_present(train_args, ["splitter_model_name"]) or "spacy",
        sentence_threshold=int(train_args.get("sentence_threshold", 50)),
        max_sentence_tokens=int(train_args.get("max_sentence_tokens", 64)),
        min_sentence_tokens_filter=int(train_args.get("min_sentence_tokens_filter", 4)),
    )
    splitter = _compat_call(create_token_based_sentence_splitter, (), splitter_kwargs)

    prompt_context = None
    loader = None
    if args.prompt:
        prompt_context = _build_prompt_context(args.prompt, tokenizer, cfg)
    else:
        loader = _prep_loaders(train_args, tokenizer, splitter)

    save_path = os.path.join(args.exp_dir, "debug", "val_inference_latest.txt") if args.save_to_debug else None
    text = rollout_from_loader(
        model,
        tokenizer,
        loader,
        save_path,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        seed=args.sample_seed,
        prompt_context=prompt_context,
    )
    print(text)
    if save_path:
        print(f"\n[Rollout also saved to {save_path}]")


if __name__ == "__main__":
    main()
