#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training loop for the dual-memory transformer model."""

import os
import sys
import json
import time
import argparse
import logging
import hashlib
import random
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from torch.cuda.amp import autocast, GradScaler

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from plotting import TrainingPlotter
from inference_rollout import rollout_from_loader


from sentence_splitter import create_token_based_sentence_splitter
from create_datasets import (
    extract_all_documents_from_files,
    split_documents_by_length_3buckets,
    process_documents_for_training,
    chunk_processed_documents,
    create_analysis_report,
    compute_full_dataset_analysis,
    write_analysis_files,
    compute_essential_dataset_stats,
    write_sample_docs,
    write_split_sentence_samples,
)


from model import SentenceTransformerConfig, SentenceTransformer, LongTermMemoryGPU, DocumentLongTermMemory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import shutil

logger = logging.getLogger("trainer")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
logger.addHandler(sh)


import math

def _to_ppl(loss_value: float) -> float:
    """
    Map a (non-negative) loss value to a 'perplexity-like' scale via exp(loss).
    We clamp the exponent to keep numbers finite in pathological cases.
    """
    try:
        return float(math.exp(min(80.0, float(loss_value))))
    except Exception:
        return float('inf')


# -----------------------------
# Dataset Caching System
# -----------------------------
class DatasetCache:
    """Robust dataset caching system"""
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / "cache_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache manifest: {e}")
        return {}

    def _save_manifest(self):
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache manifest: {e}")

    def _compute_cache_key(self, split_name: str, docs: List[Dict[str, Any]],
                          config: Dict[str, Any]) -> str:
        doc_signatures = []
        for doc in docs:
            doc_text = doc.get('text', '')
            doc_id = doc.get('doc_id', '')
            doc_hash = hashlib.md5(f"{doc_id}::{doc_text}".encode()).hexdigest()
            doc_signatures.append(doc_hash)
        doc_signatures.sort()

        config_items = {
            'split': split_name,
            'max_sentence_tokens': config.get('max_sentence_tokens', 64),
            'min_sentences': config.get('min_sentences', 10),
            'use_chunking': config.get('use_chunking', False),
            'chunk_size': config.get('chunk_size', 45),
            'chunk_overlap': config.get('chunk_overlap', 15),
            'tokenizer_vocab_size': config.get('tokenizer_vocab_size', 0),
            'special_tokens': config.get('special_tokens', []),
        }

        combined = {
            'docs': doc_signatures[:100],
            'num_docs': len(docs),
            'config': config_items
        }

        cache_key = hashlib.sha256(
            json.dumps(combined, sort_keys=True).encode()
        ).hexdigest()[:16]

        return f"{split_name}_{cache_key}"

    def get(self, split_name: str, docs: List[Dict[str, Any]],
           config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        cache_key = self._compute_cache_key(split_name, docs, config)
        if cache_key in self.manifest:
            cache_info = self.manifest[cache_key]
            cache_file = self.cache_dir / cache_info['file']
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"✓ Loaded cached {split_name} dataset: {len(data)} examples from {cache_file.name}")
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
                    del self.manifest[cache_key]
                    self._save_manifest()
        return None

    def set(self, split_name: str, docs: List[Dict[str, Any]],
           config: Dict[str, Any], data: List[Dict[str, Any]]) -> str:
        """
        Store 'data' and update manifest. Returns the cache_key used,
        so callers can write sidecar analysis files named consistently.
        """
        cache_key = self._compute_cache_key(split_name, docs, config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.manifest[cache_key] = {
                'file': cache_file.name,
                'split': split_name,
                'num_examples': len(data),
                'num_docs': len(docs),
                'config': config,
                'created': str(Path(cache_file).stat().st_mtime)
            }
            self._save_manifest()
            logger.info(f"✓ Cached {split_name} dataset: {len(data)} examples to {cache_file.name}")
            return cache_key
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")
            if cache_file.exists():
                cache_file.unlink()
            # still return a deterministic key even on failure
            return cache_key

    def clear(self, split_name: Optional[str] = None):
        if split_name:
            keys_to_remove = [k for k in self.manifest.keys() if k.startswith(split_name)]
        else:
            keys_to_remove = list(self.manifest.keys())
        for key in keys_to_remove:
            if key in self.manifest:
                cache_file = self.cache_dir / self.manifest[key]['file']
                if cache_file.exists():
                    cache_file.unlink()
                del self.manifest[key]
        self._save_manifest()
        logger.info(f"Cleared {len(keys_to_remove)} cache entries")



# -----------------------------
# Auxiliary Losses
# -----------------------------
class BagOfWordsAuxLoss(nn.Module):
    """Multi-label BoW loss with BCE."""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.projection = nn.Linear(d_model, vocab_size)

    # ---------------- existing text-based API (kept for compatibility) ----------------
    def forward(self, s_reps: torch.Tensor, texts: List[str], tokenizer) -> torch.Tensor:
        """Multi-label BCE loss using tokenizer.encode(text)."""
        if s_reps.size(0) == 0:
            return torch.tensor(0.0, device=s_reps.device)
        device = s_reps.device
        batch_size = s_reps.size(0)
        logits = self.projection(s_reps)  # [B, V]
        targets = torch.zeros(batch_size, self.vocab_size, device=device)
        for i, text in enumerate(texts):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            unique_ids = set(tid for tid in token_ids if 0 <= tid < self.vocab_size)
            for tid in unique_ids:
                targets[i, tid] = 1.0
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

    @torch.no_grad()
    def compute_accuracy(self, s_reps: torch.Tensor, texts: List[str], tokenizer) -> float:
        """Recall@k with k=#unique tokens from tokenizer.encode(text)."""
        if s_reps.size(0) == 0:
            return 0.0
        recalls = []
        logits = self.projection(s_reps)
        probs = torch.sigmoid(logits)  # [B, V]
        for i, text in enumerate(texts):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            unique_ids = list({tid for tid in token_ids if 0 <= tid < self.vocab_size})
            if not unique_ids:
                continue
            k = len(unique_ids)
            _, top_k = probs[i].topk(k)
            recalls.append(len(set(top_k.tolist()) & set(unique_ids)) / k)
        import numpy as _np
        return float(_np.mean(recalls)) if recalls else 0.0

    # ---------------- new fast path: consumes token IDs directly ----------------
    def forward_from_token_ids(self, s_reps: torch.Tensor, ids_per_example: List[List[int]]) -> torch.Tensor:
        """
        Multi-label BCE loss using token IDs (no tokenizer calls).
        ids_per_example: list of lists (per S_REP) of token IDs present in the sentence,
        typically all tokens before [EOS].
        """
        if s_reps.size(0) == 0:
            return torch.tensor(0.0, device=s_reps.device)
        device = s_reps.device
        logits = self.projection(s_reps)  # [B, V]
        targets = torch.zeros_like(logits)
        for i, tids in enumerate(ids_per_example):
            for tid in tids:
                if 0 <= tid < self.vocab_size:
                    targets[i, tid] = 1.0
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

    @torch.no_grad()
    def compute_accuracy_from_token_ids(self, s_reps: torch.Tensor, ids_per_example: List[List[int]]) -> float:
        """
        Recall@k using unique token IDs derived from the batch tensors.
        """
        if s_reps.size(0) == 0:
            return 0.0
        probs = torch.sigmoid(self.projection(s_reps))  # [B, V]
        recalls = []
        for i, tids in enumerate(ids_per_example):
            uniq = list({tid for tid in tids if 0 <= tid < self.vocab_size})
            if not uniq:
                continue
            k = len(uniq)
            topk = probs[i].topk(k).indices.tolist()
            recalls.append(len(set(topk) & set(uniq)) / k)
        import numpy as _np
        return float(_np.mean(recalls)) if recalls else 0.0


class BagOfWordsAuxLossLMHead(nn.Module):
    """
    BoW loss that routes S_REP through the model's srep_to_token -> lm_head.
    - No extra parameters are introduced.
    - Gradients flow into srep_to_token, lm_head (and, because it's tied, tok_emb).
    """
    def __init__(self, model: SentenceTransformer, vocab_size: int):
        super().__init__()
        self.model = model
        self.vocab_size = int(vocab_size)

    @staticmethod
    def _targets_from_ids(ids_per_example: List[List[int]], vocab_size: int, device) -> torch.Tensor:
        B = len(ids_per_example)
        targets = torch.zeros(B, vocab_size, device=device)
        for i, tids in enumerate(ids_per_example):
            for tid in set(tids):                 # multi-label
                if 0 <= tid < vocab_size:
                    targets[i, tid] = 1.0
        return targets

    # Optional text-based API (kept for compatibility; unused in your loop)
    def forward(self, s_reps: torch.Tensor, texts: List[str], tokenizer) -> torch.Tensor:
        if s_reps.size(0) == 0:
            return torch.tensor(0.0, device=s_reps.device)
        device = s_reps.device
        ids_per_example = []
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            ids_per_example.append([i for i in ids if 0 <= i < self.vocab_size])
        logits = self.model.lm_head(self.model.srep_to_token(s_reps))     # << use LM head
        targets = self._targets_from_ids(ids_per_example, self.vocab_size, device)
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

    def forward_from_token_ids(self, s_reps: torch.Tensor, ids_per_example: List[List[int]]) -> torch.Tensor:
        """
        Fast path that consumes token IDs directly (your training loop uses this).
        """
        if s_reps.size(0) == 0:
            return torch.tensor(0.0, device=s_reps.device)
        logits = self.model.lm_head(self.model.srep_to_token(s_reps))     # << use LM head
        targets = self._targets_from_ids(ids_per_example, self.vocab_size, s_reps.device)
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

    @torch.no_grad()
    def compute_accuracy_from_token_ids(self, s_reps: torch.Tensor, ids_per_example: List[List[int]]) -> float:
        """
        Recall@k with k = #unique target tokens in each sentence.
        """
        if s_reps.size(0) == 0:
            return 0.0
        logits = self.model.lm_head(self.model.srep_to_token(s_reps))     # << use LM head
        probs = torch.sigmoid(logits)                                      # [B, V]
        recalls = []
        for i, tids in enumerate(ids_per_example):
            uniq = list({tid for tid in tids if 0 <= tid < self.vocab_size})
            if not uniq:
                continue
            k = len(uniq)
            topk = probs[i].topk(k).indices.tolist()
            recalls.append(len(set(topk) & set(uniq)) / k)
        import numpy as _np
        return float(_np.mean(recalls)) if recalls else 0.0




class AdjacencyContrastiveLoss(nn.Module):
    """Vectorized InfoNCE with temperature for adjacent sentences within the same doc."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, s_reps: torch.Tensor, doc_ids: List[str]) -> torch.Tensor:
        if s_reps.size(0) < 2:
            return torch.tensor(0.0, device=s_reps.device)

        z = F.normalize(s_reps, p=2, dim=1)
        sim = (z @ z.t()) / self.temperature    # [n, n]
        n = sim.size(0)
        sim.fill_diagonal_(float('-inf'))                # exclude self; safe in fp16 (goes to -inf)

        # Generate symmetric adjacent pairs (prev index per doc)
        anchors, positives = [], []
        prev_idx_by_doc: Dict[str, int] = {}
        for i, d in enumerate(doc_ids):
            if d in prev_idx_by_doc:
                j = prev_idx_by_doc[d]
                anchors.append(i); positives.append(j)
                anchors.append(j); positives.append(i)
            prev_idx_by_doc[d] = i

        if not anchors:
            return torch.tensor(0.0, device=s_reps.device)

        I = torch.as_tensor(anchors, dtype=torch.long, device=s_reps.device)
        J = torch.as_tensor(positives, dtype=torch.long, device=s_reps.device)

        pos = sim[I, J].float()
        denom = torch.logsumexp(sim[I, :].float(), dim=1)
        loss = -(pos - denom).mean()
        return loss

    @torch.no_grad()
    def compute_accuracy(self, s_reps: torch.Tensor, doc_ids: List[str]) -> float:
        if s_reps.size(0) < 2:
            return 0.0

        z = F.normalize(s_reps, p=2, dim=1)
        sim = (z @ z.t()).float() / float(self.temperature)
        sim.fill_diagonal_(float('-inf'))
        prev_idx_by_doc: Dict[str, int] = {}
        correct = 0; total = 0
        for i, d in enumerate(doc_ids):
            if d in prev_idx_by_doc:
                j = prev_idx_by_doc[d]
                total += 1; correct += int(sim[i].argmax().item() == j)
                total += 1; correct += int(sim[j].argmax().item() == i)
            prev_idx_by_doc[d] = i
        return correct / max(1, total)


class TokenSentenceAlignmentLoss(nn.Module):
    """
    Alignment ("pointing") loss with three variants:
      - "tokens_only": align every valid token to S_REP (detached). Loss = mean(1 - cos)
      - "hybrid":      align q = alpha*mean(prefix tokens) + (1-alpha)*mean(STM_detached) to S_REP_detached
      - "both":        sum of the two losses above

    Returns:
      total_loss (Tensor, differentiable),
      total_acc (float in [0,1]),
      details: dict with optional keys:
          "tokens_loss", "tokens_acc", "hybrid_loss", "hybrid_acc"
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _valid_token_mask(input_ids: torch.Tensor,
                          key_pad_mask: torch.Tensor,
                          eos_id: int, srep_id: int) -> torch.Tensor:
        # Valid = not pad, not EOS, not S_REP
        not_pad  = ~key_pad_mask
        not_eos  = ~input_ids.eq(eos_id)
        not_srep = ~input_ids.eq(srep_id)
        return not_pad & not_eos & not_srep  # [B, L] boolean

    @staticmethod
    def _per_token_align(token_hiddens: torch.Tensor,
                         s_reps: torch.Tensor,
                         valid_mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Align each valid token embedding to the sentence's S_REP (detached).
        """
        if token_hiddens.numel() == 0 or s_reps.numel() == 0:
            zero = token_hiddens.new_tensor(0.0, requires_grad=True)
            return zero, 0.0

        tok = F.normalize(token_hiddens, p=2, dim=-1)  # [B, L, D]
        s   = F.normalize(s_reps.detach(), p=2, dim=-1)  # [B, D], detached
        cos = (tok * s.unsqueeze(1)).sum(dim=-1)  # [B, L]
        cos_valid = cos[valid_mask]
        if cos_valid.numel() == 0:
            zero = token_hiddens.new_tensor(0.0, requires_grad=True)
            return zero, 0.0

        loss = (1.0 - cos_valid).mean()
        acc = float(((cos_valid.clamp(-1, 1) + 1.0) * 0.5).mean().item())
        return loss, acc

    @staticmethod
    def _prefix_mean(token_hiddens: torch.Tensor,
                     valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean of valid token hiddens per row. Returns [B, D].
        """
        B, L, D = token_hiddens.shape
        mask_f = valid_mask.float().unsqueeze(-1)  # [B, L, 1]
        denom = mask_f.sum(dim=1).clamp(min=1.0)   # [B, 1]
        mean = (token_hiddens * mask_f).sum(dim=1) / denom  # [B, D]
        return mean

    @staticmethod
    def _hybrid_align(token_hiddens: torch.Tensor,
                      s_reps: torch.Tensor,
                      valid_mask: torch.Tensor,
                      stm_means: Optional[torch.Tensor],
                      alpha: float) -> Tuple[torch.Tensor, float]:
        """
        q = alpha * mean(prefix tokens) + (1 - alpha) * mean(STM_detached)
        align q to S_REP_detached via (1 - cos).
        """
        if token_hiddens.numel() == 0 or s_reps.numel() == 0:
            zero = token_hiddens.new_tensor(0.0, requires_grad=True)
            return zero, 0.0

        alpha = float(max(0.0, min(1.0, alpha)))
        prefix_mean = TokenSentenceAlignmentLoss._prefix_mean(token_hiddens, valid_mask)            # [B, D]
        stm_mean = None
        if stm_means is not None and stm_means.numel() > 0:
            # Already detached when produced by the model; detach again defensively.
            stm_mean = stm_means.detach()
        else:
            # No STM: fall back to tokens-only (alpha==1.0)
            alpha = 1.0

        if stm_mean is None:
            q = prefix_mean
        else:
            q = alpha * prefix_mean + (1.0 - alpha) * stm_mean

        q = F.normalize(q, p=2, dim=-1)                 # [B, D]
        s = F.normalize(s_reps.detach(), p=2, dim=-1)   # [B, D] detached
        cos = (q * s).sum(dim=-1)                       # [B]
        loss = (1.0 - cos).mean()
        acc = float(((cos.clamp(-1, 1) + 1.0) * 0.5).mean().item())
        return loss, acc

    def forward(self,
                token_hiddens: torch.Tensor,   # [B, L, D]
                s_reps: torch.Tensor,          # [B, D]
                input_ids: torch.Tensor,       # [B, L]
                key_pad_mask: torch.Tensor,    # [B, L], True=pad
                eos_id: int,
                srep_id: int,
                mode: str = "tokens_only",
                stm_means: Optional[torch.Tensor] = None,  # [B, D] or None (detached)
                alpha: float = 0.5) -> Tuple[torch.Tensor, float, Dict[str, float]]:

        valid = self._valid_token_mask(input_ids, key_pad_mask, eos_id, srep_id)

        total_loss = token_hiddens.new_tensor(0.0, requires_grad=True)
        total_acc = 0.0
        details: Dict[str, float] = {}

        if mode in ("tokens_only", "both"):
            tok_loss, tok_acc = self._per_token_align(token_hiddens, s_reps, valid)
            total_loss = total_loss + tok_loss
            total_acc += tok_acc
            details["tokens_loss"] = float(tok_loss.detach().item())
            details["tokens_acc"] = float(tok_acc)

        if mode in ("hybrid", "both"):
            hyb_loss, hyb_acc = self._hybrid_align(token_hiddens, s_reps, valid, stm_means, alpha)
            total_loss = total_loss + hyb_loss
            total_acc += hyb_acc
            details["hybrid_loss"] = float(hyb_loss.detach().item())
            details["hybrid_acc"] = float(hyb_acc)

        # If both, report mean accuracy; if single, it's just that component.
        if mode == "both":
            total_acc *= 0.5

        return total_loss, float(total_acc), details




# -----------------------------
# S_REP Diagnostics (Complete)
# -----------------------------
class SRepDiagnostics:
    """Track S_REP health metrics with bounded compute via reservoir sampling."""
    def __init__(self, buffer_size: int = 2048, explode_threshold: float = 5.0,
                 diversity_sample_size: int = 256, diversity_every: int = 50):
        self.prev_by_doc: Dict[str, torch.Tensor] = {}
        self.sample: List[torch.Tensor] = []   # CPU tensors
        self.buffer_size = buffer_size
        self.explode_threshold = explode_threshold

        # Diversity controls
        self.diversity_sample_size = max(32, int(diversity_sample_size))
        self.diversity_every = max(1, int(diversity_every))
        self._steps_since_diversity = 0

    def _reservoir_add(self, vecs: torch.Tensor):
        # vecs: [N, D] on device; store on CPU to keep VRAM free
        with torch.no_grad():
            for v in vecs.detach().cpu():
                if len(self.sample) < self.buffer_size:
                    self.sample.append(v)
                else:
                    import random
                    j = random.randint(0, len(self.sample))
                    if j < self.buffer_size:
                        self.sample[j] = v

    @torch.no_grad()
    def _compute_diversity_from_reservoir(self) -> Optional[float]:
        """Return 1 - mean off-diagonal cosine over a small random subset of the reservoir."""
        import random
        if len(self.sample) < 2:
            return None
        k = min(self.diversity_sample_size, len(self.sample))
        idx = random.sample(range(len(self.sample)), k)
        M = torch.stack([self.sample[i] for i in idx], dim=0)          # [k, D] on CPU
        Z = F.normalize(M, p=2, dim=1)                                  # unit-norm rows
        C = Z @ Z.t()                                                  # [k, k]
        # exclude diagonal
        diag = torch.eye(k, dtype=Z.dtype, device=Z.device)
        num = C.sum().item() - torch.trace(C).item()
        den = float(k * (k - 1))
        mean_offdiag = (num / den) if den > 0 else 0.0
        diversity = 1.0 - float(mean_offdiag)
        # Clamp to [0, 1] for safety
        return max(0.0, min(1.0, diversity))

    def track(self, s_reps: torch.Tensor, texts: List[str],
              doc_ids: List[str], raw_norms: List[float] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if s_reps.size(0) == 0:
            return metrics

        # Raw norms (pre-normalization) summary
        if raw_norms:
            import numpy as _np
            metrics['raw_norm_mean'] = float(_np.mean(raw_norms))
            metrics['raw_norm_std'] = float(_np.std(raw_norms)) if len(raw_norms) > 1 else 0.0

        # Normalized norms (hygiene)
        norms = s_reps.norm(dim=1)
        metrics['norm_mean'] = norms.mean().item()
        metrics['norm_std'] = norms.std().item() if len(norms) > 1 else 0.0

        # Maintain reservoir
        self._reservoir_add(s_reps)

        # Batch-level max similarity
        if s_reps.size(0) > 1:
            Zb = F.normalize(s_reps, p=2, dim=1)
            Mb = (Zb @ Zb.t())
            Mb.fill_diagonal_(float('-inf'))
            metrics['max_similarity'] = float(Mb.max().item())
        else:
            metrics['max_similarity'] = 0.0

        # Diversity over reservoir (throttled)
        self._steps_since_diversity += 1
        if self._steps_since_diversity >= self.diversity_every:
            self._steps_since_diversity = 0
            div = self._compute_diversity_from_reservoir()
            if div is not None:
                metrics['diversity'] = div

        # Adjacent coherence (previous in same doc stream)
        adj_sims = []
        for s_rep, doc_id in zip(s_reps, doc_ids):
            if doc_id in self.prev_by_doc:
                prev = self.prev_by_doc[doc_id]
                sim = F.cosine_similarity(s_rep.unsqueeze(0).cpu(), prev.unsqueeze(0)).item()
                adj_sims.append(sim)
            self.prev_by_doc[doc_id] = s_rep.detach().cpu()
        import numpy as _np
        metrics['adjacent_coherence'] = float(_np.mean(adj_sims)) if adj_sims else 0.0

        return metrics


def gate_regularizer_loss(model) -> torch.Tensor:
    """
    Very gentle L2 regularizer that nudges both memory/self gates toward 1.0.
    Stronger on layer 0; extremely small elsewhere (safe for 'starved' gates).
    Only applies to gates that require_grad to avoid adding constant loss terms.
    """
    cfg = model.cfg
    if not getattr(cfg, "enable_gate_regularizer", True):
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

    w0 = float(getattr(cfg, "gate_reg_first_layer", 1e-2))
    w  = float(getattr(cfg, "gate_reg_other_layers", 1e-3))
    use_self = bool(getattr(cfg, "gate_reg_include_self", True))
    use_mem  = bool(getattr(cfg, "gate_reg_include_memory", True))

    loss: Optional[torch.Tensor] = None
    device = next(model.parameters()).device

    for i, block in enumerate(model.blocks):
        wi = w0 if i == 0 else w
        if wi <= 0:
            continue
        if use_mem and getattr(block, "memory_gate", None) is not None and block.memory_gate.requires_grad:
            t = wi * (block.memory_gate - 1.0).pow(2)
            loss = t if loss is None else loss + t
        if use_self and getattr(block, "self_gate", None) is not None and block.self_gate.requires_grad:
            t = wi * (block.self_gate - 1.0).pow(2)
            loss = t if loss is None else loss + t

    if loss is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss


# -----------------------------
# Cadence helper (batch-boundary aware)
# -----------------------------
def _batch_will_hit_cadence(start_step: int, batch_docs: List[Dict[str, Any]], every: int) -> bool:
    """True if this batch (which yields exactly T_max steps) will include a modulo step."""
    if every is None or every <= 0:
        return False
    # The generator emits one sentence-step per t in [0, T_max-1]
    t_max = max((len(d.get("sentences", [])) for d in batch_docs), default=0)
    if t_max <= 0:
        return False
    rem = start_step % every
    to_next = (every - rem) % every
    return to_next < t_max

def _batch_cadence_hit_step(start_step: int, batch_docs: List[Dict[str, Any]], every: int) -> Optional[int]:
    """
    If this batch crosses a cadence boundary 'every', return the exact global
    step inside this batch when it happens; otherwise return None.
    We use this to stamp sparse series so they align with metrics['train_steps'].
    """
    if every is None or every <= 0:
        return None
    # How many sentence-steps will this batch produce?
    t_max = max((len(d.get("sentences", [])) for d in batch_docs), default=0)
    if t_max <= 0:
        return None
    rem = start_step % every
    to_next = (every - rem) % every  # 0 means boundary at batch start
    if to_next < t_max:
        return start_step + to_next
    return None


# -----------------------------
# Parameter-level component grad norms (zero-overhead)
# -----------------------------
import math
from typing import Iterable, Dict, Tuple, List

def _grad_l2_of(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of gradients over a set of parameters, without allocating new tensors."""
    sq = 0.0
    for p in params:
        if p is None:
            continue
        g = getattr(p, "grad", None)
        if g is None:
            continue
        if g.is_sparse:
            g = g.coalesce().values()
        # use float() to avoid dtype surprises under AMP
        sq += float(g.detach().float().pow(2).sum().item())
    return math.sqrt(sq) if sq > 0.0 else 0.0


def collect_param_component_grad_norms(model) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Return (avg_across_blocks, block0_only, last_block_only) dictionaries with keys:
      'cross_attn', 'self_attn', 'ffn', 'layernorm'
    Values are L2 grad norms of the corresponding parameter sets.
    """
    comps = ("cross_attn", "self_attn", "ffn", "layernorm")
    per_block: List[Dict[str, float]] = []

    for blk in model.blocks:
        # Cross-attention params
        ca = []
        mha = blk.cross_attn
        ca.extend([getattr(mha, "in_proj_weight", None),
                   getattr(mha, "in_proj_bias", None),
                   getattr(mha.out_proj, "weight", None),
                   getattr(mha.out_proj, "bias", None)])

        # Self-attention params
        sa = []
        mha2 = blk.self_attn
        sa.extend([getattr(mha2, "in_proj_weight", None),
                   getattr(mha2, "in_proj_bias", None),
                   getattr(mha2.out_proj, "weight", None),
                   getattr(mha2.out_proj, "bias", None)])

        # FFN params (both Linear layers inside the Sequential)
        ff = []
        for m in blk.mlp:
            if isinstance(m, torch.nn.Linear):
                ff.append(m.weight)
                if m.bias is not None:
                    ff.append(m.bias)

        # LayerNorms in this block (pre-LN)
        ln = [blk.ln_mem.weight, blk.ln_mem.bias,
              blk.ln_self.weight, blk.ln_self.bias,
              blk.ln_ffn.weight, blk.ln_ffn.bias]

        norms = {
            "cross_attn": _grad_l2_of(ca),
            "self_attn":  _grad_l2_of(sa),
            "ffn":        _grad_l2_of(ff),
            "layernorm":  _grad_l2_of(ln),
        }
        per_block.append(norms)

    if not per_block:
        zero = {k: 0.0 for k in comps}
        return zero, zero, zero

    # Average across blocks (arithmetic mean of per-block norms)
    n = len(per_block)
    avg = {k: sum(b[k] for b in per_block) / float(n) for k in comps}
    blk0 = per_block[0]
    blklast = per_block[-1]
    return avg, blk0, blklast


# -----------------------------
# Helper Functions for Metrics
# -----------------------------
def update_metrics_with_srep_analytics(metrics: Dict, srep_diagnostics, srep_embs: torch.Tensor,
                                      srep_texts: List[str], doc_ids: List[str],
                                      raw_norms: List[float] = None):
    """Update metrics with S_REP analytics including raw norms"""
    if srep_embs.numel() == 0:
        return

    srep_metrics = srep_diagnostics.track(srep_embs, srep_texts, doc_ids, raw_norms)

    # Step stamp for any sparse S_REP series recorded on this logging tick
    step_for_srep = None
    if "train_steps" in metrics and metrics["train_steps"]:
        step_for_srep = int(metrics["train_steps"][-1])

    for key, value in srep_metrics.items():
        metric_key = f"srep_{key}"
        metrics.setdefault(metric_key, []).append(value)

        # Diversity is computed only every N calls; keep its true X so we can align when plotting.
        if step_for_srep is not None and key == "diversity":
            metrics.setdefault("srep_diversity_steps", []).append(step_for_srep)



def update_metrics_with_ltm_stats(metrics: Dict, ltm_stats_by_block: Dict):
    """Update metrics with LTM statistics"""
    if not ltm_stats_by_block:
        return

    all_first, all_topk, all_valid, all_used = [], [], [], []

    for block_idx, stats_data in ltm_stats_by_block.items():
        if not stats_data:
            continue

        # Handle both single dict and list of dicts
        if isinstance(stats_data, list):
            stats_list = stats_data
        else:
            stats_list = [stats_data]

        for stats in stats_list:
            if isinstance(stats, dict):
                all_first.append(stats.get("first_sim", 0.0))
                all_topk.append(stats.get("mean_topk", 0.0))
                all_valid.append(stats.get("mean_valid", 0.0))
                all_used.append(stats.get("used_ratio", 0.0))

    if all_first:
        metrics.setdefault("ltm_first_sim", []).append(float(np.mean(all_first)))
    else:
        metrics.setdefault("ltm_first_sim", []).append(0.0)

    if all_topk:
        metrics.setdefault("ltm_mean_topk", []).append(float(np.mean(all_topk)))
    else:
        metrics.setdefault("ltm_mean_topk", []).append(0.0)

    if all_valid:
        metrics.setdefault("ltm_mean_valid", []).append(float(np.mean(all_valid)))
    else:
        metrics.setdefault("ltm_mean_valid", []).append(0.0)

    if all_used:
        metrics.setdefault("ltm_used_ratio", []).append(float(np.mean(all_used)))
    else:
        metrics.setdefault("ltm_used_ratio", []).append(0.0)


def update_metrics_with_query_comparison(metrics: Dict, query_comparison: Dict):
    """Track query mode comparison metrics"""
    if not query_comparison:
        return

    tokens_first, hybrid_first = [], []
    tokens_topk, hybrid_topk = [], []

    for layer_idx, stats_data in query_comparison.items():
        if not stats_data:
            continue

        # Handle both single dict and list of dicts
        if isinstance(stats_data, list):
            stats_list = stats_data
        else:
            stats_list = [stats_data]

        for stats in stats_list:
            if isinstance(stats, dict):
                tokens_first.append(stats.get('tokens_first_sim', 0))
                hybrid_first.append(stats.get('hybrid_first_sim', 0))
                tokens_topk.append(stats.get('tokens_mean_topk', 0))
                hybrid_topk.append(stats.get('hybrid_mean_topk', 0))

    if tokens_first:
        metrics.setdefault("query_tokens_first_sim", []).append(float(np.mean(tokens_first)))
    if hybrid_first:
        metrics.setdefault("query_hybrid_first_sim", []).append(float(np.mean(hybrid_first)))
    if tokens_topk:
        metrics.setdefault("query_tokens_mean_topk", []).append(float(np.mean(tokens_topk)))
    if hybrid_topk:
        metrics.setdefault("query_hybrid_mean_topk", []).append(float(np.mean(hybrid_topk)))


# -----------------------------
# Warmup Schedule (Stepwise)
# -----------------------------
class WarmupSchedule:
    """
    4 phases of equal length (phase_steps), with context-dropout levels:
      levels[0] -> levels[1] -> levels[2] -> levels[3]
    LTM is OFF before `ltm_start_phase_index`, and ON after (optionally ramped).
    Aux multiplier follows 3.0 / 2.0 / 1.5 / 1.0 across the 4 phases.
    """
    def __init__(self,
                 dropout_levels=(0.0, 0.03, 0.06, 0.10),
                 phase_steps: int = 20000,
                 ltm_start_phase_index: int = 2,
                 ltm_ramp_steps: int = 0):
        assert len(dropout_levels) == 4, "Expect exactly 4 dropout levels"
        self.levels = [float(x) for x in dropout_levels]
        self.phase_steps = int(phase_steps)
        self.ltm_start_phase_index = int(ltm_start_phase_index)
        self.ltm_ramp_steps = max(0, int(ltm_ramp_steps))

    def _phase_index(self, step: int) -> int:
        return min(3, int(step // self.phase_steps))

    def get_context_dropout(self, step: int) -> float:
        return self.levels[self._phase_index(step)]

    def get_memory_weight(self, step: int) -> float:
        phase = self._phase_index(step)
        if phase < self.ltm_start_phase_index:
            return 0.0
        if self.ltm_ramp_steps <= 0:
            return 1.0
        start = self.ltm_start_phase_index * self.phase_steps
        progress = min(1.0, max(0.0, (step - start) / float(self.ltm_ramp_steps)))
        return float(progress)

    def get_aux_weight_multiplier(self, step: int) -> float:
        phase = self._phase_index(step)
        if phase == 0:
            return 3.0
        elif phase == 1:
            return 2.0
        elif phase == 2:
            return 1.5
        else:
            return 1.0

    def get_phase_name(self, step: int) -> str:
        return "STM" if self._phase_index(step) < self.ltm_start_phase_index else "STM+LTM"



# -----------------------------
# Experiment Manager
# -----------------------------
class ExperimentManager:
    def __init__(self, base_dir: str, exp_name: str):
        self.base = os.path.join(base_dir, exp_name)
        self.paths = {
            "logs": os.path.join(self.base, "logs"),
            "plots": os.path.join(self.base, "plots"),
            "cache": os.path.join(self.base, "cache"),
            "checkpoints": os.path.join(self.base, "checkpoints"),
            "debug": os.path.join(self.base, "debug"),
        }
        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.paths["logs"], "experiment.log"))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    def path(self, key: str) -> str:
        return self.paths[key]


# -----------------------------
# Dataset
# -----------------------------
class DocumentDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

from torch.utils.data import Sampler

class DocumentSequentialSampler(Sampler[int]):
    """
    Yields dataset indices such that:
      • All chunks of the same document appear consecutively and in-order (chunk_0, chunk_1, ...).
      • The overall document order is shuffled (and can be re-shuffled each epoch).
    Use with DataLoader(batch_size=1, shuffle=False) in document LTM mode to stream docs sequentially.
    """
    def __init__(self, dataset: DocumentDataset, shuffle_docs: bool = True, seed: int = 42, reshuffle_each_epoch: bool = True):
        self.dataset = dataset
        self.shuffle_docs = bool(shuffle_docs)
        self.seed = int(seed)
        self.reshuffle_each_epoch = bool(reshuffle_each_epoch)
        self._epoch = 0
        self._order: List[int] = []
        self._build_order()

    def _build_order(self):
        from collections import defaultdict
        import random

        by_doc = defaultdict(list)  # doc_id -> list[(chunk_idx, idx)]
        for idx, ex in enumerate(self.dataset.items):
            doc_id = ex.get("original_doc_id", ex.get("doc_id", f"doc_{idx}"))
            ci = ex.get("chunk_info", None)
            cidx = int(ci.get("chunk_idx", 0)) if isinstance(ci, dict) else 0
            by_doc[doc_id].append((cidx, idx))

        docs = list(by_doc.keys())
        if self.shuffle_docs:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(docs)

        order: List[int] = []
        for d in docs:
            seq = sorted(by_doc[d], key=lambda t: t[0])
            order.extend([i for _, i in seq])
        self._order = order

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        if self.reshuffle_each_epoch:
            self._build_order()

    def __iter__(self):
        return iter(self._order)

    def __len__(self):
        return len(self._order)


def collate_documents(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch



# -----------------------------
# Evaluation (with per-component losses + perplexities)
# -----------------------------
@torch.no_grad()
def evaluate(
    model: SentenceTransformer,
    ltm: Optional[LongTermMemoryGPU],
    val_loader: DataLoader,
    device: torch.device,
    memory_weight: float,
    aux_losses: Optional[Dict[str, nn.Module]] = None,
    args: Optional[argparse.Namespace] = None,
    aux_weight_mult: float = 1.0,
    # Debug writing controls
    debug_dump_dir: Optional[str] = None,
    debug_dump_n_sentences: int = 5,
    debug_dump_n_docs: int = 5,
    debug_dump_topk: int = 5,
    global_step: Optional[int] = None,
) -> Tuple[float, float, Dict, Dict]:
    """
    Returns:
      - avg_main_loss_per_token (float)
      - token accuracy (float)
      - avg_attn_metrics (dict)
      - eval_summary (dict)
    Side effect (optional):
      - Overwrites .../debug/val_retrieval_latest.txt with up to `debug_dump_n_sentences` samples.
    """
    model.eval()
    amp_enabled = (device.type == "cuda")
    from collections import defaultdict, deque
    import random

    # Collect last N sentences per doc for debugging
    per_doc_buffers = defaultdict(lambda: deque(maxlen=int(max(1, debug_dump_n_sentences))))

    # Use a fresh LTM for evaluation in document-scope mode
    if isinstance(ltm, DocumentLongTermMemory):
        eval_ltm = DocumentLongTermMemory(ltm.embedding_dim, device)
    else:
        eval_ltm = ltm

    total_ce_sum, total_tokens, total_correct = 0.0, 0, 0
    total_ce_sum_no_eos, total_tokens_no_eos = 0.0, 0
    total_eos_ce_sum, total_eos_tokens = 0.0, 0
    comp_sums = {'main': 0.0, 'norm': 0.0, 'bow': 0.0, 'adjacency': 0.0, 'alignment': 0.0}
    total_step_loss_sum = 0.0
    step_count = 0

    have_bow = bool(aux_losses and ('bow' in aux_losses))
    have_adj = bool(aux_losses and ('adjacency' in aux_losses))
    have_align = bool(aux_losses and ('alignment' in aux_losses))
    alignment_mode_eval = (
        getattr(args, "alignment_variant", "tokens_only") if args is not None else "tokens_only"
    )
    need_alignment_features_eval = have_align
    need_stm_means_eval = need_alignment_features_eval and alignment_mode_eval in ("hybrid", "both")

    collect_debug_dump = bool(debug_dump_dir)
    debug_samples: List[Dict[str, Any]] = []
    max_debug = max(0, int(debug_dump_n_sentences))
    topk = max(1, int(debug_dump_topk))

    for batch_docs in tqdm(val_loader, desc="Evaluating"):
        # reset per-document store at the beginning of each document
        if isinstance(eval_ltm, DocumentLongTermMemory):
            for ex in batch_docs:
                doc_key = ex.get("original_doc_id", ex.get("doc_id"))
                ci = ex.get("chunk_info", {})
                chunk_idx = int(ci.get("chunk_idx", 0)) if isinstance(ci, dict) else 0
                if chunk_idx == 0:
                    eval_ltm.reset_document(doc_key)

        srep_embs_all, srep_texts_all, srep_doc_ids_all = [], [], []

        with autocast(enabled=amp_enabled):
            for step_rec in model.iter_document_steps(
                batch_docs,
                ltm=eval_ltm,
                warmup_weight=memory_weight,
                collect_debug=False,
                collect_ltm_debug=collect_debug_dump,
                ltm_debug_topk=topk,
                need_alignment_features=need_alignment_features_eval,
                need_stm_means=need_stm_means_eval,
            ):
                logits = step_rec["logits"]
                ids = step_rec["input_ids"]
                key_pad_mask = step_rec["key_pad_mask"]

                # accumulate S_REPs for eval LTM construction
                if step_rec["srep_embs"].numel() > 0:
                    srep_embs_all.append(step_rec["srep_embs"])
                    srep_texts_all.extend(step_rec["srep_texts"])
                    srep_doc_ids_all.extend(step_rec["srep_doc_ids"])

                # MAIN (per-token)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = ids[:, 1:].contiguous()
                valid = ~key_pad_mask[:, 1:]
                valid &= ~shift_labels.eq(model.srep_id)

                if valid.any():
                    labels = shift_labels.masked_fill(~valid, -100)
                    loss_flat = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="none"
                    ).view_as(shift_labels)
                    ce_sum = float(loss_flat[valid].sum().item())
                    tokens = int(valid.sum().item())
                    total_ce_sum += ce_sum
                    total_tokens += tokens
                    preds = shift_logits.argmax(dim=-1)
                    total_correct += (preds.eq(shift_labels) & valid).sum().item()

                    valid_no_eos = valid & ~shift_labels.eq(model.eos_id)
                    eos_mask = valid & shift_labels.eq(model.eos_id)
                    tokens_no_eos = int(valid_no_eos.sum().item())
                    eos_tokens = int(eos_mask.sum().item())
                    if tokens_no_eos > 0:
                        total_ce_sum_no_eos += float(loss_flat[valid_no_eos].sum().item())
                        total_tokens_no_eos += tokens_no_eos
                    if eos_tokens > 0:
                        total_eos_ce_sum += float(loss_flat[eos_mask].sum().item())
                        total_eos_tokens += eos_tokens

                    main_mean = ce_sum / max(1, tokens)
                else:
                    main_mean = 0.0

                # Norm regularization (already weighted inside step)
                norm_loss = float(step_rec["srep_norm_reg_loss"].item())

                # BoW (weighted)
                bow_w = args.bow_loss_weight * float(aux_weight_mult) if (args is not None) else 1.0
                bow_loss_w = 0.0
                if have_bow:
                    s_reps_mem = step_rec["srep_embs"]
                    if s_reps_mem.numel() > 0:
                        ids_per_example: List[List[int]] = []
                        for i in range(ids.size(0)):
                            srep_pos = (ids[i] == model.srep_id).nonzero(as_tuple=False).squeeze(-1)
                            if srep_pos.numel() == 0:
                                continue
                            T = int(srep_pos[0].item()) - 1
                            T = max(0, T)
                            ids_per_example.append(ids[i, :T].tolist())
                        if ids_per_example:
                            bow_base = aux_losses['bow'].forward_from_token_ids(s_reps_mem, ids_per_example).item()
                            bow_loss_w = bow_w * bow_base

                # Adjacency (weighted)
                adj_loss_w = 0.0
                if have_adj:
                    adj_sreps = step_rec.get("adjacency_sreps", None)
                    adj_doc_ids = step_rec.get("adjacency_doc_ids", None)
                    if isinstance(adj_sreps, torch.Tensor) and adj_sreps.size(0) > 1 and isinstance(adj_doc_ids, list):
                        adj_base = aux_losses['adjacency'](adj_sreps, adj_doc_ids).item()
                        adj_loss_w = (args.adjacency_loss_weight if args is not None else 1.0) * adj_base

                # Alignment (weighted) — USE TOKEN-SPACE S_REP and obey alignment_start_step
                align_loss_w = 0.0
                if have_align and ("token_hiddens" in step_rec):
                    allow_align_now = True
                    if args is not None and global_step is not None:
                        allow_align_now = (global_step >= int(getattr(args, "alignment_start_step", 0)))
                    if allow_align_now:
                        # Prefer token-space projection; fall back if missing
                        s_reps_token = step_rec.get("srep_embs_token", None)
                        s_reps = s_reps_token if (isinstance(s_reps_token, torch.Tensor) and s_reps_token.numel() > 0) \
                                 else step_rec["srep_embs"]
                        # Only compute if dimensions match token_hiddens
                        if s_reps.numel() > 0 and s_reps.size(-1) == step_rec["token_hiddens"].size(-1):
                            align_base, _acc, _details = aux_losses['alignment'](
                                token_hiddens=step_rec["token_hiddens"],
                                s_reps=s_reps,                               # <-- token space when available
                                input_ids=ids,
                                key_pad_mask=key_pad_mask,
                                eos_id=model.eos_id,
                                srep_id=model.srep_id,
                                mode=(args.alignment_variant if args is not None else "tokens_only"),
                                stm_means=step_rec.get("stm_means", None),
                                alpha=(float(args.alignment_alpha) if args is not None else 0.5),
                            )
                            align_loss_w = (args.alignment_loss_weight if args is not None else 1.0) * float(align_base.item())

                # Aggregate per-step
                step_total = float(main_mean) + float(norm_loss) + float(bow_loss_w) + float(adj_loss_w) + float(align_loss_w)
                total_step_loss_sum += step_total
                step_count += 1

                comp_sums['main'] += float(main_mean)
                comp_sums['norm'] += float(norm_loss)
                comp_sums['bow'] += float(bow_loss_w)
                comp_sums['adjacency'] += float(adj_loss_w)
                comp_sums['alignment'] += float(align_loss_w)

                # Debug dump collection
                if collect_debug_dump:
                    vis = step_rec.get("visible_texts", [])
                    ltm_dbg = step_rec.get("ltm_debug_by_block", {})
                    s_texts = step_rec.get("srep_texts", [])
                    s_docs  = step_rec.get("srep_doc_ids", [])
                    B = len(s_texts)
                    for bi in range(B):
                        doc_id = s_docs[bi] if bi < len(s_docs) else ""
                        per_block = {}
                        for bidx, rows in ltm_dbg.items():
                            if isinstance(rows, list) and bi < len(rows):
                                per_block[int(bidx)] = rows[bi]
                        per_doc_buffers[doc_id].append({
                            'doc_id': doc_id,
                            'full_sentence': s_texts[bi] if bi < len(s_texts) else "",
                            'visible_text': vis[bi] if bi < len(vis) else "",
                            'per_block': per_block
                        })

                # Update eval LTM immediately
                if isinstance(eval_ltm, DocumentLongTermMemory) and step_rec["srep_embs"].numel() > 0:
                    eval_ltm.add_or_update_batch(
                        step_rec["srep_texts"], step_rec["srep_embs"], step_rec["srep_doc_ids"]
                    )

    # Aggregate outputs
    avg_main_loss = total_ce_sum / total_tokens if total_tokens > 0 else float('inf')
    avg_main_loss_no_eos = total_ce_sum_no_eos / total_tokens_no_eos if total_tokens_no_eos > 0 else 0.0
    avg_main_loss_eos = total_eos_ce_sum / total_eos_tokens if total_eos_tokens > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    denom = max(1, step_count)
    components_loss_per_step = {
        'main': comp_sums['main'] / denom,
        'norm': comp_sums['norm'] / denom,
    }
    if have_bow:
        components_loss_per_step['bow'] = comp_sums['bow'] / denom
    if have_adj:
        components_loss_per_step['adjacency'] = comp_sums['adjacency'] / denom
    if have_align:
        components_loss_per_step['alignment'] = comp_sums['alignment'] / denom

    total_loss_per_step = total_step_loss_sum / denom if denom > 0 else float('inf')
    components_ppl = {k: _to_ppl(v) for k, v in components_loss_per_step.items()}
    eval_summary = {
        'total_loss_per_step': float(total_loss_per_step),
        'components_loss_per_step': {k: float(v) for k, v in components_loss_per_step.items()},
        'perplexities': {
            'main_token': float(_to_ppl(avg_main_loss)),
            'main_token_no_eos': float(_to_ppl(avg_main_loss_no_eos))
        },
        'losses': {
            'main_token': float(avg_main_loss),
            'main_token_no_eos': float(avg_main_loss_no_eos),
            'main_token_eos': float(avg_main_loss_eos)
        },
        'loss_sums': {
            'main_token': float(total_ce_sum),
            'main_token_no_eos': float(total_ce_sum_no_eos),
            'main_token_eos': float(total_eos_ce_sum)
        },
        'token_counts': {
            'total': int(total_tokens),
            'no_eos': int(total_tokens_no_eos),
            'eos': int(total_eos_tokens)
        }
    }

    avg_attn_metrics = {}

    # Debug file (latest only)
    if collect_debug_dump:
        all_docs = [d for d, buf in per_doc_buffers.items() if len(buf) > 0]
        if all_docs:
            random.shuffle(all_docs)
            pick = all_docs[:max(1, int(debug_dump_n_docs))]
            debug_samples = []
            for d in pick:
                debug_samples.extend(list(per_doc_buffers[d])[-int(max(1, debug_dump_n_sentences)):])
        else:
            debug_samples = []
        latest_path = os.path.join(debug_dump_dir, "val_retrieval_latest.txt")
        write_validation_retrieval_debug(latest_path, debug_samples, model.cfg)
        logger.info(f"Validation retrieval debug (latest) written to: {latest_path}")

    return avg_main_loss, accuracy, avg_attn_metrics, eval_summary



class STMGradProbe:
    """
    Every-batch gradient probe for STM S_REPs.

    Behavior:
      • On EACH sentence step, call add_refs(global_step, stm_grad_refs) if present.
      • After loss.backward(), call measure_and_reset() to compute the mean grad-norm
        for each step we queued during this forward pass; logs one (step, value) per step.
      • A hard cap (limit) keeps the cost negligible.

    Logged metric keys to append:
      - metrics["stm_srep_grad_steps"] : int step index (global_step you pass in)
      - metrics["stm_srep_grad"]       : float mean L2 grad norm
    """
    def __init__(self, limit: int = 100):
        self.limit = max(1, int(limit))
        # queue holds (step_idx, [Tensor,...]) for this micro-batch accumulation window
        self._queue: List[Tuple[int, List[torch.Tensor]]] = []

    def add_refs(self, step: int, refs: List[torch.Tensor]) -> None:
        if not refs:
            return
        picked: List[torch.Tensor] = []
        for r in refs:
            if r is None:
                continue
            # We only care about tensors that participate in autograd this step
            if not r.requires_grad and getattr(r, "grad_fn", None) is None:
                continue
            picked.append(r)
            if len(picked) >= self.limit:
                break
        if picked:
            self._queue.append((int(step), picked))

    def measure_and_reset(self) -> List[Tuple[int, float]]:
        if not self._queue:
            return []
        results: List[Tuple[int, float]] = []
        import numpy as _np
        for step_idx, refs in self._queue:
            grads = [t.grad.norm().item() for t in refs if (t.grad is not None)]
            if not grads:   # skip writing empty points
                continue
            mean_norm = float(_np.mean(grads))
            results.append((step_idx, mean_norm))
        self._queue.clear()
        return results


class SplitManifest:
    """
    Persist and reuse the exact train/val doc_id lists so dataset cache keys match across runs.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _doc_signature(doc: Dict[str, Any]) -> str:
        # Stable signature: doc_id + md5(document text or concatenated sentence_texts)
        doc_id = str(doc.get("doc_id", ""))
        text = doc.get("text", "")
        if not text:
            texts = doc.get("sentence_texts", [])
            if texts:
                text = "\n".join(texts)
        h_txt = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        return hashlib.md5(f"{doc_id}::{h_txt}".encode()).hexdigest()

    def compute_key(self, docs: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        # Use a capped, sorted sample of signatures to keep key small but stable
        sigs = [self._doc_signature(d) for d in docs]
        sigs.sort()
        sigs = sigs[:500]
        payload = {"num_docs": len(docs), "sigs": sigs, "params": params}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.cache_dir / f"split_manifest_{key}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if data.get("key") == key:
                return data
        except Exception as e:
            logger.warning(f"Failed to load split manifest: {e}")
        return None

    def save(self, key: str, data: Dict[str, Any]) -> str:
        path = self.cache_dir / f"split_manifest_{key}.json"
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write split manifest: {e}")
        return str(path)


def write_validation_retrieval_debug(
    out_path: str,
    samples: List[Dict[str, Any]],
    model_cfg: SentenceTransformerConfig
) -> str:
    """
    Write a human-readable report of LTM retrievals for a handful of validation sentences.

    Each sample item:
      {
        'doc_id': str,
        'full_sentence': str,
        'visible_text': str,
        'per_block': {
           block_idx: {
              'ltm_used_pre': bool or None,
              'ltm_used_post': bool,
              'ltm_early_skip': bool,
              'items': [
                 {'rank': int,'sim': float,'text': str,'initial_valid': bool,'final_status': str,'selected': bool}, ...
              ]
           }, ...
        }
      }
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines: List[str] = []

    lines.append("="*88)
    lines.append(f"VALIDATION RETRIEVAL DEBUG")
    lines.append("="*88)
    lines.append(f"LTM settings: use_long_term_memory={model_cfg.use_long_term_memory} | "
                 f"top_k={model_cfg.ltm_top_k} | min_sim={model_cfg.ltm_min_sim:.3f} | "
                 f"min_matches={model_cfg.ltm_min_matches} | query_mode={model_cfg.ltm_query_mode}")
    lines.append(f"STM capacity: {model_cfg.max_sentences_in_short_term} | "
                 f"no_ltm_for_first_k_blocks={model_cfg.no_ltm_for_first_k_blocks}")
    lines.append("")

    if not samples:
        lines.append("(No samples collected or LTM disabled.)")
    else:
        for si, sample in enumerate(samples, start=1):
            lines.append("-"*88)
            lines.append(f"SAMPLE {si}")
            lines.append(f"Doc ID: {sample.get('doc_id','')}")
            lines.append("Full sentence:")
            lines.append(sample.get('full_sentence', ''))
            lines.append("")
            lines.append("Visible to model (query prefix):")
            lines.append(sample.get('visible_text', ''))
            lines.append("")

            per_block = sample.get('per_block', {})
            if not per_block:
                lines.append("(No LTM retrieval details for this sample.)")
                lines.append("")
                continue

            for bidx in sorted(per_block.keys()):
                bd = per_block[bidx]
                lines.append(f"[Block {bidx}] "
                             f"used_pre={bd.get('ltm_used_pre')} | "
                             f"early_skip={bd.get('ltm_early_skip')} | "
                             f"used_post={bd.get('ltm_used_post')}")
                items = bd.get('items', [])
                if not items:
                    lines.append("  (no candidates)")
                    lines.append("")
                    continue
                for it in items:
                    status = it.get('final_status', 'unknown')
                    sim = it.get('sim', 0.0)
                    rk  = it.get('rank', 0)
                    initial_valid = it.get('initial_valid', False)
                    lines.append(f"  #{rk:02d}  sim={sim:.4f}  "
                                 f"initial_valid={initial_valid}  status={status}")
                    lines.append(f"      {it.get('text','')}")
                lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path



# -----------------------------
# Main Training Function
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # -----------------------------
    # Data
    # -----------------------------
    ap.add_argument("--data_folder", type=str, required=True)
    ap.add_argument("--extensions", nargs="+", default=[".jsonl", ".txt", ".json", ".train", ".parquet", ".pq"])
    ap.add_argument("--sample_docs_per_split", type=int, default=10)
    ap.add_argument("--min_sentences_per_article", type=int, default=2)
    ap.add_argument("--max_chars_per_doc", type=int, default=-1)
    ap.add_argument("--num_data_workers", type=int, default=0,
                    help="Number of DataLoader workers. Default keeps loading on the main process.")
    ap.add_argument("--prefetch_factor", type=int, default=2,
                    help="Prefetch factor per worker when num_data_workers > 0.")

    # -----------------------------
    # Experiment
    # -----------------------------
    ap.add_argument("--output_dir", type=str, default="experiments21_speedup_srepEQdm_memGates")
    ap.add_argument("--exp_name", type=str, default="run")

    # -----------------------------
    # Model architecture
    # -----------------------------
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--context_length", type=int, default=66)

    # -----------------------------
    # Memory settings
    # -----------------------------
    ap.add_argument("--max_sentences_in_short_term", type=int, default=15)
    ap.add_argument("--use_long_term_memory", action="store_true")
    ap.add_argument("--no_ltm_for_first_k_blocks", type=int, default=4)
    ap.add_argument("--ltm_top_k", type=int, default=5)
    ap.add_argument("--ltm_min_sim", type=float, default=0.3)
    ap.add_argument("--ltm_min_matches", type=int, default=2)
    ap.add_argument("--ltm_query_mode", type=str, default="hybrid",
                    choices=["tokens_only", "hybrid", "both"])

    ap.add_argument("--ltm_scope", type=str, default="global",
                choices=["global", "document"],
                help="Scope of long-term memory: global (default) or per-document (resets at each document).")
    # Seed for doc order shuffling in document-scope training
    ap.add_argument("--seed", type=int, default=42)

    # -----------------------------
    # STM positional encoding
    # -----------------------------
    ap.add_argument("--use_stm_positional", type=bool, default=True)
    ap.add_argument("--stm_positional_weight", type=float, default=1.0)
    ap.add_argument("--memory_gate_init", type=float, default=1.0)  # NO residual alpha anywhere

    # -----------------------------
    # Auxiliary losses (NO diversity loss)
    # -----------------------------
    ap.add_argument("--use_bow_loss", action="store_true")
    ap.add_argument("--bow_use_lm_head", action="store_true",
                help="Route BoW through srep_to_token -> lm_head instead of a private projection.")

    ap.add_argument("--bow_loss_weight", type=float, default=0.15)

    ap.add_argument("--use_adjacency_loss", action="store_true")
    ap.add_argument("--adjacency_loss_weight", type=float, default=0.1)
    ap.add_argument("--adjacency_temperature", type=float, default=0.07)
    ap.add_argument("--adjacency_start_step", type=int, default=10000,
                help="Global step after which adjacency contrastive loss is enabled")

    ap.add_argument("--use_alignment_loss", action="store_true")
    ap.add_argument("--alignment_loss_weight", type=float, default=0.1)
    ap.add_argument("--alignment_start_step", type=int, default=10000,
                    help="Enable alignment (pointing) loss after this global step")
    ap.add_argument("--alignment_variant", type=str, default="tokens_only",
                choices=["tokens_only", "hybrid", "both"])
    ap.add_argument("--alignment_alpha", type=float, default=0.5)

    ap.add_argument("--context_dropout", type=float, default=0.1)


    # -----------------------------
    # S_REP settings
    # -----------------------------
    ap.add_argument("--srep_dropout", type=float, default=0.1)
    ap.add_argument("--srep_norm_target", type=float, default=1.0)
    ap.add_argument("--srep_norm_margin", type=float, default=0.1)
    ap.add_argument("--srep_norm_reg_weight", type=float, default=0.01)

    # -----------------------------
    # Debug modes
    # -----------------------------
    ap.add_argument("--debug_no_memory", action="store_true")
    ap.add_argument("--debug_stm_only", action="store_true")

    # -----------------------------
    # Gradient capture
    # -----------------------------
    ap.add_argument("--stm_grad_every", type=int, default=5000,
                    help="Collect STM S_REP grad refs every N global steps (0 disables).")
    ap.add_argument("--stm_grad_limit", type=int, default=100,
                    help="Max number of STM S_REP refs to keep per measurement window.")

    # Component grad capture (Cross-Attn / Self-Attn / FFN / LayerNorm)
    ap.add_argument("--comp_grad_every", type=int, default=5000,
                    help="Collect component gradient refs every N global steps (0 disables).")
    ap.add_argument("--comp_grad_limit", type=int, default=100,
                    help="Max per-component refs to keep per measurement window.")
    ap.add_argument("--comp_grad_doc_limit", type=int, default=2,
                    help="Limit number of batch rows (documents) whose component refs are sampled when measuring.")


    # -----------------------------
    # Splitter
    # -----------------------------
    ap.add_argument("--max_sentence_tokens", type=int, default=64)
    ap.add_argument("--min_sentence_tokens", type=int, default=3)
    ap.add_argument("--use_model_splitter", type=bool, default=True)
    ap.add_argument("--splitter_model_name", type=str, default="sat-3l-sm")
    ap.add_argument("--sentence_threshold", type=float, default=0.15)

    # -----------------------------
    # Training
    # -----------------------------
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=20000)
    ap.add_argument("--plot_every", type=int, default=10000)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)

    ap.add_argument("--epoch_window", type=int, default=20000,
                help="Number of steps to display in the zoomed plots (interpreted as ~last epoch).")


    # --- S_REP / memory space / head / pooling ---
    ap.add_argument("--srep_dim", type=int, default=None,
                    help="Sentence representation dimension. Defaults to d_model when omitted.")
    ap.add_argument("--srep_head_depth", type=int, default=1)
    ap.add_argument("--srep_head_mult", type=int, default=1)
    ap.add_argument("--srep_head_activation", type=str, default="gelu", choices=["gelu", "swiglu"])
    ap.add_argument("--use_attentive_pool", type=bool, default=False)
    ap.add_argument("--attn_pool_n_queries", type=int, default=1)   # learned queries in d_model space

    # --- 4-phase context-dropout schedule (20k each by default) ---
    ap.add_argument("--dropout_phase_steps", type=int, default=20000)
    ap.add_argument("--dropout_levels", type=float, nargs=4, default=[0.0, 0.03, 0.06, 0.10])
    ap.add_argument("--dropout_activation_delta", type=float, default=1.0,
                    help="Enable scheduled context dropout once val perplexity improves less than this threshold between evals.")
    ap.add_argument("--ltm_start_phase_index", type=int, default=2)  # 0-based: 2 => phase 3/4
    ap.add_argument("--ltm_ramp_steps", type=int, default=0)         # 0 = hard switch

    ap.add_argument("--attn_dropout", type=float, default=0.1)

    # -----------------------------
    # Document chunking
    # -----------------------------
    ap.add_argument("--use_chunking", type=bool, default=True)
    ap.add_argument("--chunk_size", type=int, default=90)
    ap.add_argument("--chunk_overlap", type=int, default=15)

    # -----------------------------
    # Cache management
    # -----------------------------
    ap.add_argument("--clear_cache", action="store_true")
    ap.add_argument("--no_cache", action="store_true")

    args = ap.parse_args()

    # Experiment manager & logging
    em = ExperimentManager(args.output_dir, args.exp_name)
    if args.srep_dim is None:
        args.srep_dim = args.d_model

    if (not args.use_long_term_memory) and args.ltm_start_phase_index != 0:
        logger.info("Long-term memory disabled → forcing ltm_start_phase_index=0 so STM is active from step 0.")
        args.ltm_start_phase_index = 0

    logger.info("Starting training with configuration:")
    logger.info(json.dumps(vars(args), indent=2))

    # -----------------------------
    # Cache init & clear (if requested)
    # -----------------------------
    cache = DatasetCache(em.path("cache"))
    if args.clear_cache:
        cache.clear()
        logger.info("Cache cleared")

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {
        "pad_token": "[PAD]",
        "eos_token": "[EOS]",
        "additional_special_tokens": ["[S_REP]"]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    final_vocab_size = len(tokenizer)
    logger.info(f"Tokenizer updated: +{num_added} special tokens, final vocab = {final_vocab_size}")

    # -----------------------------
    # Splitter
    # -----------------------------
    splitter = create_token_based_sentence_splitter(
        tokenizer=tokenizer,
        use_model=args.use_model_splitter,
        model_name=args.splitter_model_name,
        sentence_threshold=args.sentence_threshold,
        max_sentence_tokens=args.max_sentence_tokens,
        min_sentence_tokens=args.min_sentence_tokens,
    )

    # -----------------------------
    # Load data files
    # -----------------------------
    files = []
    for root, _, fns in os.walk(args.data_folder):
        for fn in fns:
            if any(fn.lower().endswith(ext) for ext in args.extensions):
                files.append(os.path.join(root, fn))
    if not files:
        raise ValueError("No data files found.")
    files.sort()
    logger.info(f"Found {len(files)} data files")

    # -----------------------------
    # Extract docs & split
    # -----------------------------
    # -----------------------------
    # Load data files (with optional explicit train/val directories)
    # -----------------------------
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
    has_train = os.path.isdir(train_dir)
    has_val   = os.path.isdir(val_dir)

    explicit_splits = has_train and has_val
    if explicit_splits:
        logger.info(f"✓ Found explicit split directories: {train_dir}  and  {val_dir}")
        train_files = _collect_files(train_dir, args.extensions)
        val_files   = _collect_files(val_dir,   args.extensions)
        if not train_files:
            raise ValueError(f"No data files found under {train_dir}")
        if not val_files:
            raise ValueError(f"No data files found under {val_dir}")

        # Build docs directly per split
        train_docs = extract_all_documents_from_files(
            train_files, splitter, use_document_boundaries=True,
            max_chars_per_doc=args.max_chars_per_doc
        )
        val_docs = extract_all_documents_from_files(
            val_files, splitter, use_document_boundaries=True,
            max_chars_per_doc=args.max_chars_per_doc
        )

        # Small readable samples
        # Small readable samples (sentence-split view: one sentence per line)
        write_split_sentence_samples(train_docs, em.path("cache"), "train", k=args.sample_docs_per_split, splitter=splitter)
        write_split_sentence_samples(val_docs,   em.path("cache"), "val",   k=args.sample_docs_per_split, splitter=splitter)

    else:
        missing = []
        if not has_train: missing.append("train/")
        if not has_val:   missing.append("val/")
        logger.info(
            "Explicit split directories not found (missing: %s). "
            "Falling back to internal, document-balanced split.",
            ", ".join(missing) if missing else "(none)"
        )

        files = _collect_files(args.data_folder, args.extensions)
        if not files:
            raise ValueError("No data files found.")
        logger.info(f"Found {len(files)} data files")

        docs = extract_all_documents_from_files(
            files, splitter, use_document_boundaries=True,
            max_chars_per_doc=args.max_chars_per_doc
        )

        # Persist and reuse the doc-balanced split via SplitManifest
        split_params = {
            "method": "length_3buckets",
            "train_val_split": 0.9,
            "seed": 42,
            "min_sentences_per_document": args.min_sentences_per_article,
            "max_sentence_tokens": args.max_sentence_tokens,
            "use_model_splitter": args.use_model_splitter,
            "splitter_model_name": args.splitter_model_name,
            "sentence_threshold": args.sentence_threshold,
        }

        split_manifest = SplitManifest(em.path("cache"))
        split_key = split_manifest.compute_key(docs, split_params)
        saved = split_manifest.load(split_key)

        if saved is not None:
            train_ids_set = set(saved.get("train_doc_ids", []))
            val_ids_set   = set(saved.get("val_doc_ids", []))
            id_to_doc = {str(d.get("doc_id", "")): d for d in docs}
            train_docs = [id_to_doc[i] for i in saved["train_doc_ids"] if i in id_to_doc]
            val_docs   = [id_to_doc[i] for i in saved["val_doc_ids"]   if i in id_to_doc]
            if len(train_docs) + len(val_docs) != len(train_ids_set) + len(val_ids_set):
                logger.warning("Split manifest references unknown docs; recomputing split.")
                saved = None

        if saved is None:
            from create_datasets import split_documents_by_length_3buckets
            train_docs, val_docs = split_documents_by_length_3buckets(
                docs,
                tokenizer=tokenizer,
                splitter=splitter,
                min_sentences_per_document=args.min_sentences_per_article,
                train_val_split=0.9,
                seed=42,
            )
            payload = {
                "key": split_key,
                "params": split_params,
                "train_doc_ids": [str(d.get("doc_id", "")) for d in train_docs],
                "val_doc_ids":   [str(d.get("doc_id", "")) for d in val_docs],
                "counts": {"train": len(train_docs), "val": len(val_docs)},
            }
            split_manifest.save(split_key, payload)
            logger.info(f"Saved split manifest for reuse (train={len(train_docs)}, val={len(val_docs)}).")
        else:
            logger.info(f"Reused split manifest (train={len(train_docs)}, val={len(val_docs)}).")

        # Write small human-readable samples (first 10 docs) for quick inspection
        try:
            train_smpl = write_sample_docs(train_docs, os.path.join(em.path("cache"), "train_sample_docs.txt"), k=10)
            val_smpl   = write_sample_docs(val_docs,   os.path.join(em.path("cache"), "val_sample_docs.txt"),   k=10)
            logger.info(f"Wrote sample docs → {train_smpl} and {val_smpl}")
        except Exception as e:
            logger.warning(f"Failed to write sample docs: {e}")

        try:
            write_split_sentence_samples(train_docs, em.path("cache"), "train", k=args.sample_docs_per_split, splitter=splitter)
            write_split_sentence_samples(val_docs,   em.path("cache"), "val",   k=args.sample_docs_per_split, splitter=splitter)
        except Exception as e:
            logger.warning(f"Failed to write sentence-split samples: {e}")



    # -----------------------------
    # Cache config
    # -----------------------------
    cache_config = {
        'max_sentence_tokens': args.max_sentence_tokens,
        'min_sentences': args.min_sentences_per_article,
        'use_chunking': args.use_chunking,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'tokenizer_vocab_size': final_vocab_size,
        'special_tokens': [tokenizer.pad_token, tokenizer.eos_token, '[S_REP]'],
        'min_sentence_tokens': args.min_sentence_tokens,
    }


    # -----------------------------
    # Process datasets (+chunking) with caching
    # -----------------------------
    train_built_fresh = False
    val_built_fresh = False

    if args.no_cache:
        # Build both splits fresh (no cache)
        train_examples = process_documents_for_training(
            train_docs, tokenizer, splitter,
            max_sentence_tokens=args.max_sentence_tokens,
            min_sentences_per_document=args.min_sentences_per_article,
            min_sentence_tokens_filter=args.min_sentence_tokens,
        )
        if args.use_chunking:
            train_examples = chunk_processed_documents(
                train_examples, chunk_size=args.chunk_size,
                overlap_size=args.chunk_overlap, split_name="train"
            )

        val_examples = process_documents_for_training(
            val_docs, tokenizer, splitter,
            max_sentence_tokens=args.max_sentence_tokens,
            min_sentences_per_document=args.min_sentences_per_article,
            min_sentence_tokens_filter=args.min_sentence_tokens,
        )
        if args.use_chunking:
            val_examples = chunk_processed_documents(
                val_examples, chunk_size=args.chunk_size,
                overlap_size=args.chunk_overlap, split_name="val"
            )

        train_built_fresh = True
        val_built_fresh = True
    else:
        # Try cache; if missing, build and then cache
        train_examples = cache.get("train", train_docs, cache_config)
        if train_examples is None:
            train_examples = process_documents_for_training(
                train_docs, tokenizer, splitter,
                max_sentence_tokens=args.max_sentence_tokens,
                min_sentences_per_document=args.min_sentences_per_article,
                min_sentence_tokens_filter=args.min_sentence_tokens,
            )
            if args.use_chunking:
                train_examples = chunk_processed_documents(
                    train_examples, chunk_size=args.chunk_size,
                    overlap_size=args.chunk_overlap, split_name="train"
                )
            cache.set("train", train_docs, cache_config, train_examples)
            train_built_fresh = True

        val_examples = cache.get("val", val_docs, cache_config)
        if val_examples is None:
            val_examples = process_documents_for_training(
                val_docs, tokenizer, splitter,
                max_sentence_tokens=args.max_sentence_tokens,
                min_sentences_per_document=args.min_sentences_per_article,
                min_sentence_tokens_filter=args.min_sentence_tokens,
            )
            if args.use_chunking:
                val_examples = chunk_processed_documents(
                    val_examples, chunk_size=args.chunk_size,
                    overlap_size=args.chunk_overlap, split_name="val"
                )
            cache.set("val", val_docs, cache_config, val_examples)
            val_built_fresh = True

    # ----- ANALYSIS KEYS (for filenames) -----
    if args.no_cache:
        train_cache_key = "nocache"
        val_cache_key   = "nocache"
    else:
        train_cache_key = cache._compute_cache_key("train", train_docs, cache_config)
        val_cache_key   = cache._compute_cache_key("val",   val_docs,   cache_config)

    # ----- WRITE ANALYSIS FILES (ONLY when freshly built) -----
    # Train
    if train_built_fresh:
        # Full dataset analysis
        analysis_train_full = compute_full_dataset_analysis("train", train_examples)
        write_analysis_files(em.path("cache"), "train", train_cache_key, analysis_train_full)

        # Essentials analysis (exactly the requested counters)
        train_pre = process_documents_for_training(
            train_docs, tokenizer, splitter,
            max_sentence_tokens=args.max_sentence_tokens,
            min_sentences_per_document=args.min_sentences_per_article,
        )
        analysis_train_ess = compute_essential_dataset_stats("train", train_pre, train_examples)
        write_analysis_files(em.path("cache"), "train.essentials", train_cache_key, analysis_train_ess)
    else:
        logger.info("Train dataset loaded from cache — skipping analysis recompute.")

    # Val
    if val_built_fresh:
        # Full dataset analysis
        analysis_val_full = compute_full_dataset_analysis("val", val_examples)
        write_analysis_files(em.path("cache"), "val", val_cache_key, analysis_val_full)

        # Essentials analysis (exactly the requested counters)
        val_pre = process_documents_for_training(
            val_docs, tokenizer, splitter,
            max_sentence_tokens=args.max_sentence_tokens,
            min_sentences_per_document=args.min_sentences_per_article,
        )
        analysis_val_ess = compute_essential_dataset_stats("val", val_pre, val_examples)
        write_analysis_files(em.path("cache"), "val.essentials", val_cache_key, analysis_val_ess)
    else:
        logger.info("Val dataset loaded from cache — skipping analysis recompute.")

    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_ds = DocumentDataset(train_examples)
    val_ds = DocumentDataset(val_examples)

    loader_common = dict(
        batch_size=args.batch_size,
        collate_fn=collate_documents,
        pin_memory=True,
        num_workers=max(0, args.num_data_workers)
    )
    if loader_common["num_workers"] > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = max(1, args.prefetch_factor)

    if args.ltm_scope == "document":
        # Stream each document sequentially, one 90-sentence chunk per batch.
        doc_sampler = DocumentSequentialSampler(train_ds, shuffle_docs=True, seed=args.seed, reshuffle_each_epoch=True)
        train_loader_kwargs = dict(loader_common)
        train_loader_kwargs.update(dict(shuffle=False, sampler=doc_sampler))
        train_loader = DataLoader(train_ds, **train_loader_kwargs)
    else:
        train_loader_kwargs = dict(loader_common)
        train_loader_kwargs.update(dict(shuffle=True))
        train_loader = DataLoader(train_ds, **train_loader_kwargs)

    if args.ltm_scope == "document":
        # Stream each validation document sequentially (chunk_0 -> chunk_1 -> ...).
        # No reshuffle for eval to keep it deterministic.
        val_sampler = DocumentSequentialSampler(
            val_ds,
            shuffle_docs=False,
            seed=args.seed,
            reshuffle_each_epoch=False
        )
        val_loader_kwargs = dict(loader_common)
        val_loader_kwargs.update(dict(shuffle=False, sampler=val_sampler))
        val_loader = DataLoader(val_ds, **val_loader_kwargs)
    else:
        val_loader_kwargs = dict(loader_common)
        val_loader_kwargs.update(dict(shuffle=False))
        val_loader = DataLoader(val_ds, **val_loader_kwargs)


    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # -----------------------------
    # No memory_setup
    # -----------------------------

    if args.debug_no_memory:
        args.use_bow_loss = False
        args.use_alignment_loss = False
        args.use_adjacency_loss = False
        args.context_dropout = 0.0
        args.srep_norm_reg_weight = 0.0

    # -----------------------------
    # Model config & init
    # -----------------------------
    cfg = SentenceTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffw_mult=4,  # keep your default unless you expose it
        max_position_embeddings=args.context_length,
        vocab_size=final_vocab_size,

        # --- S_REP space (also memory K/V dim) + head/pooling knobs ---
        srep_dim=args.srep_dim,
        srep_head_depth=args.srep_head_depth,
        srep_head_mult=args.srep_head_mult,
        srep_head_activation=args.srep_head_activation,
        use_attentive_pool=args.use_attentive_pool,
        attn_pool_n_queries=args.attn_pool_n_queries,

        max_sentences_in_short_term=args.max_sentences_in_short_term,
        use_long_term_memory=args.use_long_term_memory,
        no_ltm_for_first_k_blocks=args.no_ltm_for_first_k_blocks,
        ltm_top_k=args.ltm_top_k,
        ltm_min_sim=args.ltm_min_sim,
        ltm_min_matches=args.ltm_min_matches,
        ltm_query_mode=args.ltm_query_mode,

        use_stm_positional=args.use_stm_positional,
        stm_positional_weight=args.stm_positional_weight,
        memory_gate_init=args.memory_gate_init,

        # Set max context dropout (actual per-step value is scheduled)
        context_dropout=args.dropout_levels[-1],

        attn_dropout=args.attn_dropout,
        srep_dropout=args.srep_dropout,

        debug_no_memory=args.debug_no_memory,
        debug_stm_only=args.debug_stm_only,
        max_sentence_tokens=args.max_sentence_tokens,
        srep_norm_target=args.srep_norm_target,
        srep_norm_margin=args.srep_norm_margin,
        srep_norm_reg_weight=args.srep_norm_reg_weight,
    )


    model = SentenceTransformer(cfg, tokenizer).to(device)


    # -----------------------------
    # Aux losses
    # -----------------------------
    aux_losses = {}
    aux_params = []
    if args.use_bow_loss:
        if args.bow_use_lm_head:
            # Uses shared model weights; no extra parameters to add to optimizer.
            aux_losses['bow'] = BagOfWordsAuxLossLMHead(model, final_vocab_size).to(device)
            # DO NOT extend aux_params (this module has no parameters)
        else:
            aux_losses['bow'] = BagOfWordsAuxLoss(cfg.srep_dim, final_vocab_size).to(device)
            aux_params.extend(aux_losses['bow'].parameters())

    if args.use_adjacency_loss:
        aux_losses['adjacency'] = AdjacencyContrastiveLoss(temperature=args.adjacency_temperature).to(device)
    if args.use_alignment_loss:
        aux_losses['alignment'] = TokenSentenceAlignmentLoss().to(device)

    alignment_mode = getattr(args, "alignment_variant", "tokens_only")
    need_alignment_features = 'alignment' in aux_losses
    need_stm_means = need_alignment_features and alignment_mode in ("hybrid", "both")

    # -----------------------------
    # Parameter count logging
    # -----------------------------
    def _count_params(m: nn.Module):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    model_total, model_trainable = _count_params(model)
    aux_total, aux_trainable = 0, 0
    if aux_params:
        _aux_param_set = {id(p): p for p in aux_params}  # unique
        aux_total = sum(p.numel() for p in _aux_param_set.values())
        aux_trainable = sum(p.numel() for p in _aux_param_set.values() if p.requires_grad)

    logger.info("─"*72)
    logger.info(f"Model parameters: total={model_total:,} | trainable={model_trainable:,}")
    if aux_total > 0:
        logger.info(f"Aux modules parameters: total={aux_total:,} | trainable={aux_trainable:,}")
        logger.info(f"Grand total (model+aux): {(model_total+aux_total):,} | "
                    f"trainable={(model_trainable+aux_trainable):,}")
    logger.info("─"*72)

    if cfg.use_long_term_memory:
        if args.ltm_scope == "document":
            ltm = DocumentLongTermMemory(cfg.srep_dim, device)  # srep_dim
        else:
            ltm = LongTermMemoryGPU(cfg.srep_dim, device)       # srep_dim
    else:
        ltm = None


    # -----------------------------
    # Optimizer & scheduler
    # -----------------------------
    def build_param_groups(m: nn.Module, aux_params: List[nn.Parameter], wd: float):
        decay, no_decay = [], []

        for name, p in m.named_parameters():
            if not p.requires_grad:
                # Completely exclude frozen params (e.g., gates in no-memory, cross-attn when disabled)
                continue

            nlow = name.lower()
            is_bias      = name.endswith(".bias")
            is_norm      = ("layernorm" in nlow) or ("ln_" in nlow) or (".ln" in nlow) or (".norm" in nlow)
            is_gate      = ("gate" in nlow)  # memory_gate, self_gate

            # 1D tensors (norm/gates/bias) and all explicit norm/gate/bias params -> no weight decay
            if p.ndim == 1 or is_bias or is_norm or is_gate:
                no_decay.append(p)
            else:
                decay.append(p)

        groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        if aux_params:
            # Auxiliary heads/logit projections usually without decay
            groups.append({"params": [p for p in aux_params if p.requires_grad], "weight_decay": 0.0})

        return groups

    param_groups = build_param_groups(model, aux_params, wd=0.01)

    optim = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))

    all_params = [p for g in param_groups for p in g["params"]]


    total_steps = max(1, len(train_loader) * args.epochs)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - args.warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # -----------------------------
    # Diagnostics & plotting
    # -----------------------------
    warmup_schedule = WarmupSchedule(
        dropout_levels=tuple(args.dropout_levels),
        phase_steps=args.dropout_phase_steps,
        ltm_start_phase_index=args.ltm_start_phase_index,
        ltm_ramp_steps=args.ltm_ramp_steps
    )
    srep_diagnostics = SRepDiagnostics(buffer_size=4000, explode_threshold=5.0)
    plotter = TrainingPlotter(em.path("plots"), n_layers=cfg.n_layers, stm_capacity=args.max_sentences_in_short_term)

    amp_enabled = (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)


    # -----------------------------
    # Training loop
    # -----------------------------
    probe = STMGradProbe(limit=getattr(args, "stm_grad_limit", 100))


    metrics = defaultdict(list)
    global_step = 0
    best_val_loss_no_eos = float("inf")
    dropout_enabled = (args.dropout_activation_delta <= 0.0)
    prev_val_perplexity = None
    dropout_activation_step: Optional[int] = None
    t0 = time.time()

    # ---- Auto-resume from best checkpoint if present ----
    resume_dir = os.path.join(em.path("checkpoints"), "best")
    model_path = os.path.join(resume_dir, "model.pt")
    meta_path  = os.path.join(resume_dir, "meta.json")
    train_state_path = os.path.join(resume_dir, "train_state.pt")
    ltm_path   = os.path.join(resume_dir, "ltm.pt")

    if os.path.isdir(resume_dir) and os.path.exists(model_path):
        try:
            # 1) Model weights
            state_dict = torch.load(model_path, map_location=device)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                logger.warning(f"Strict load failed ({e}); trying non-strict load.")
                model.load_state_dict(state_dict, strict=False)
            logger.info(f"✓ Loaded model weights from {model_path}")

            # 2) Meta (global_step, best loss)
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                best_val_loss_no_eos = float(meta.get(
                    "best_val_loss_no_eos",
                    meta.get("best_val_loss", best_val_loss_no_eos)
                ))
                global_step = int(meta.get("global_step", global_step))
                dropout_enabled = bool(meta.get("dropout_enabled", dropout_enabled))
                prev_val_perplexity = meta.get("prev_val_perplexity", prev_val_perplexity)
                if prev_val_perplexity is not None:
                    prev_val_perplexity = float(prev_val_perplexity)
                dropout_activation_step = meta.get("dropout_activation_step", dropout_activation_step)
                if dropout_activation_step is not None:
                    dropout_activation_step = int(dropout_activation_step)
                logger.info(
                    f"✓ Resume meta: global_step={global_step}, best_val_loss_no_eos={best_val_loss_no_eos:.6f}"
                )

            # 3) Trainer state (optimizer/scheduler/scaler) — only if present
            if os.path.exists(train_state_path):
                ts = torch.load(train_state_path, map_location=device)
                try:
                    optim.load_state_dict(ts["optimizer"])
                    sched.load_state_dict(ts["scheduler"])
                    if "scaler" in ts:
                        scaler.load_state_dict(ts["scaler"])
                    logger.info("✓ Restored optimizer/scheduler/AMP scaler state.")
                except Exception as e:
                    logger.warning(f"Could not restore optimizer/scheduler/scaler: {e}")

            # 4) Global LTM (if used previously)
            if os.path.exists(ltm_path) and isinstance(ltm, LongTermMemoryGPU):
                ltm_state = torch.load(ltm_path, map_location="cpu")
                ltm.texts = ltm_state.get("texts", [])
                ltm.sentence_to_idx = ltm_state.get("sentence_to_idx", {})
                emb = ltm_state.get("embeddings", None)
                if emb is not None:
                    ltm.embeddings = emb.to(device)
                logger.info(f"✓ Restored global LTM ({len(ltm.texts)} items).")

            logger.info("✅ Resuming training from the best checkpoint.")
        except Exception as e:
            logger.warning(f"Failed to resume from best checkpoint: {e}. Starting fresh.")
    else:
        logger.info("No best checkpoint found — starting from scratch.")


    metrics["n_layers"] = cfg.n_layers

    for epoch in range(args.epochs):
        model.train()
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        if args.ltm_scope == "document":
            train_loader.sampler.set_epoch(epoch)

        # Running aggregates
        running_main_loss_sum, running_tokens, running_correct = 0.0, 0, 0
        running_main_step_loss_sum, running_main_step_count = 0.0, 0  # per-step CE average
        running_main_loss_sum_no_eos, running_tokens_no_eos = 0.0, 0
        running_main_step_loss_sum_no_eos, running_main_step_count_no_eos = 0.0, 0
        running_eos_loss_sum, running_eos_count = 0.0, 0


        running_bow_acc, running_adj_acc = 0.0, 0.0
        running_bow_count, running_adj_count = 0, 0

        running_normreg_loss = 0.0
        running_normreg_step_sum, running_normreg_step_count = 0.0, 0
        running_bow_step_sum, running_bow_step_count = 0.0, 0
        running_adj_step_sum, running_adj_step_count = 0.0, 0
        running_align_step_sum, running_align_step_count = 0.0, 0
        running_align_acc, running_align_count = 0.0, 0
        running_gate_reg_step_sum, running_gate_reg_step_count = 0.0, 0


        actual_stm_sizes_batch = []

        for batch_idx, batch_docs in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Doc-scope LTM should start empty at each document boundary
            if isinstance(ltm, DocumentLongTermMemory):
                for ex in batch_docs:
                    doc_key = ex.get("original_doc_id", ex.get("doc_id"))
                    ci = ex.get("chunk_info", {})
                    chunk_idx = int(ci.get("chunk_idx", 0)) if isinstance(ci, dict) else 0
                    if chunk_idx == 0:    # first chunk of this document
                        ltm.reset_document(doc_key)

            # Decide once per batch if we will probe (cheap): only when the batch crosses the cadence boundary
            stm_probe_step  = _batch_cadence_hit_step(global_step, batch_docs, getattr(args, "stm_grad_every", 0))
            comp_probe_step = _batch_cadence_hit_step(global_step, batch_docs, getattr(args, "comp_grad_every", 0))

            will_probe_stm  = (stm_probe_step  is not None)
            will_probe_comp = (comp_probe_step is not None)

            step_losses = []
            srep_embs_all, srep_texts_all, srep_doc_ids_all = [], [], []
            srep_raw_norms_all = []
            batch_ltm_stats, batch_query_comparison = {}, {}
            actual_stm_sizes_batch = []

            with autocast(enabled=amp_enabled):
                memory_weight_now = warmup_schedule.get_memory_weight(global_step)
                context_dropout_now = (
                    warmup_schedule.get_context_dropout(global_step)
                    if dropout_enabled else 0.0
                )
                for sent_idx, step_rec in enumerate(model.iter_document_steps(
                    batch_docs,
                    ltm=ltm,
                    warmup_weight=memory_weight_now,
                    collect_debug=False,

                    # gradient probes (as you already had)
                    collect_stm_grad_refs=will_probe_stm,
                    stm_grad_limit=args.stm_grad_limit,
                    stm_grad_every=args.stm_grad_every,

                    collect_comp_grad_refs=False,
                    comp_grad_limit=args.comp_grad_limit,
                    comp_grad_every=args.comp_grad_every,
                    comp_grad_doc_limit=args.comp_grad_doc_limit,

                    global_step_start=global_step,

                    # ---- scheduled context-dropout (applies only to in-sentence masking path) ----
                    context_dropout_now=context_dropout_now,
                    need_alignment_features=need_alignment_features,
                    need_stm_means=need_stm_means,
                )):
                    logits = step_rec["logits"]
                    ids = step_rec["input_ids"]
                    key_pad_mask = step_rec["key_pad_mask"]

                    srep_embs_step = step_rec["srep_embs"]
                    srep_texts_step = step_rec["srep_texts"]
                    srep_doc_ids_step = step_rec["srep_doc_ids"]
                    srep_norm_reg_loss = step_rec["srep_norm_reg_loss"]
                    srep_raw_norms = step_rec.get("srep_raw_norms", [])

                    ltm_stats_step = step_rec.get("ltm_stats_by_block", {})
                    query_comp_step = step_rec.get("query_comparison", {})
                    actual_stm_sizes_step = step_rec.get("actual_stm_sizes", [])

                    # Optional: accumulate STM grad probe refs
                    if "stm_grad_refs" in step_rec:
                        probe.add_refs(step_rec.get("stm_grad_step", global_step), step_rec["stm_grad_refs"])


                    # Adjacency pairs (from model)
                    adj_sreps_step = step_rec.get("adjacency_sreps", srep_embs_step)
                    adj_doc_ids_step = step_rec.get("adjacency_doc_ids", srep_doc_ids_step)
                    num_adj_pairs_step = step_rec.get("num_adjacent_pairs", 0)

                    actual_stm_sizes_batch.extend(actual_stm_sizes_step)
                    srep_raw_norms_all.extend(srep_raw_norms)

                    # Main next-token loss (mean per step)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = ids[:, 1:].contiguous()
                    valid = ~key_pad_mask[:, 1:]
                    valid &= ~shift_labels.eq(model.srep_id)

                    if valid.any():
                        labels = shift_labels.masked_fill(~valid, -100)
                        step_main_loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                            reduction="mean"
                        )
                        step_token_count = valid.sum().item()
                        valid_no_eos = valid & ~shift_labels.eq(model.eos_id)
                        eos_mask = valid & shift_labels.eq(model.eos_id)
                        step_token_count_no_eos = valid_no_eos.sum().item()
                        eos_token_count = eos_mask.sum().item()
                        with torch.no_grad():
                            loss_flat = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                labels.view(-1),
                                ignore_index=-100,
                                reduction="none"
                            ).view_as(shift_labels)
                            loss_sum_no_eos = float(loss_flat[valid_no_eos].sum().item())
                            loss_sum_eos = float(loss_flat[eos_mask].sum().item()) if eos_token_count > 0 else 0.0
                            preds = shift_logits.argmax(dim=-1)
                            correct = (preds.eq(shift_labels) & valid).sum().item()
                            running_correct += correct
                            running_tokens += step_token_count
                            running_main_loss_sum += step_main_loss.item() * step_token_count
                            running_main_step_loss_sum += step_main_loss.item()
                            running_main_step_count += 1
                            if step_token_count_no_eos > 0:
                                running_main_loss_sum_no_eos += loss_sum_no_eos
                                running_tokens_no_eos += step_token_count_no_eos
                                running_main_step_loss_sum_no_eos += loss_sum_no_eos / step_token_count_no_eos
                                running_main_step_count_no_eos += 1
                            if eos_token_count > 0:
                                running_eos_loss_sum += loss_sum_eos
                                running_eos_count += int(eos_token_count)
                    else:
                        step_main_loss = torch.tensor(0.0, device=device, requires_grad=True)

                    # ---------- AUX LOSSES ----------
                    step_aux_losses = {}

                    # BoW targets from token IDs
                    bow_targets_ids: List[List[int]] = []
                    if srep_embs_step.numel() > 0:
                        for i in range(ids.size(0)):
                            srep_pos = (ids[i] == model.srep_id).nonzero(as_tuple=False).squeeze(-1)
                            if srep_pos.numel() == 0:
                                continue
                            T = int(srep_pos[0].item()) - 1  # exclude EOS at srep_pos-1
                            T = max(0, T)
                            bow_targets_ids.append(ids[i, :T].tolist())

                    if 'bow' in aux_losses and srep_embs_step.numel() > 0:
                        bow_loss_val = aux_losses['bow'].forward_from_token_ids(srep_embs_step, bow_targets_ids)
                        bow_loss_weighted = bow_loss_val * args.bow_loss_weight * warmup_schedule.get_aux_weight_multiplier(global_step)
                        step_aux_losses['bow'] = bow_loss_weighted
                        running_bow_step_sum += float(bow_loss_weighted.item())
                        running_bow_step_count += 1
                        with torch.no_grad():
                            bow_acc = aux_losses['bow'].compute_accuracy_from_token_ids(srep_embs_step, bow_targets_ids)
                            running_bow_acc += bow_acc * max(1, len(bow_targets_ids))
                            running_bow_count += max(1, len(bow_targets_ids))

                    # Adjacency loss applies after 'adjacency_start_step'
                    if 'adjacency' in aux_losses and adj_sreps_step.size(0) > 1:
                        if global_step >= args.adjacency_start_step:
                            adj_loss_val = aux_losses['adjacency'](adj_sreps_step, adj_doc_ids_step)
                            adj_loss_weighted = adj_loss_val * args.adjacency_loss_weight
                            step_aux_losses['adjacency'] = adj_loss_weighted

                            running_adj_step_sum += float(adj_loss_weighted.item())
                            running_adj_step_count += 1
                            with torch.no_grad():
                                adj_acc = aux_losses['adjacency'].compute_accuracy(adj_sreps_step, adj_doc_ids_step)
                                running_adj_acc += adj_acc
                                running_adj_count += 1

                            metrics.setdefault("adjacency_pairs", []).append(num_adj_pairs_step)
                            metrics.setdefault("adjacency_pairs_steps", []).append(global_step)
                        else:
                            step_aux_losses['adjacency'] = torch.tensor(0.0, device=device)

                    # ---------- Alignment / Pointing loss (tokens -> detached S_REP) ----------
                    if 'alignment' in aux_losses and srep_embs_step.numel() > 0 and "token_hiddens" in step_rec:
                        if global_step >= args.alignment_start_step:
                            align_loss_val, align_acc, _align_details = aux_losses['alignment'](
                                token_hiddens=step_rec["token_hiddens"],
                                s_reps=step_rec.get("srep_embs_token", step_rec["srep_embs"]),  # token space preferred
                                input_ids=ids,
                                key_pad_mask=key_pad_mask,
                                eos_id=model.eos_id,
                                srep_id=model.srep_id,
                                mode=(args.alignment_variant if args is not None else "tokens_only"),
                                stm_means=step_rec.get("stm_means", None),   # d_model
                                alpha=(float(args.alignment_alpha) if args is not None else 0.5),
                            )
                            align_loss_weighted = align_loss_val * args.alignment_loss_weight
                            step_aux_losses['alignment'] = align_loss_weighted

                            running_align_step_sum += float(align_loss_weighted.item())
                            running_align_step_count += 1
                            running_align_acc += float(align_acc)
                            running_align_count += 1
                        else:
                            step_aux_losses['alignment'] = torch.tensor(0.0, device=device)


                    # Norm reg per-step accounting
                    running_normreg_loss += float(srep_norm_reg_loss.item())
                    running_normreg_step_sum += float(srep_norm_reg_loss.item())
                    running_normreg_step_count += 1

                    # Total step loss (under autocast)
                    step_loss = step_main_loss + srep_norm_reg_loss
                    for aux_loss in step_aux_losses.values():
                        step_loss = step_loss + aux_loss
                    step_losses.append(step_loss)

                    # Collect S_REPs for LTM updates and diagnostics
                    if srep_embs_step.numel() > 0:
                        srep_embs_all.append(srep_embs_step)
                        srep_texts_all.extend(srep_texts_step)
                        srep_doc_ids_all.extend(srep_doc_ids_step)

                    # Aggregate LTM and query comparison stats
                    for block_idx, stats in ltm_stats_step.items():
                        batch_ltm_stats.setdefault(block_idx, []).append(stats)
                    for block_idx, stats in query_comp_step.items():
                        batch_query_comparison.setdefault(block_idx, []).append(stats)

                    # ------------- periodic metrics -------------
                    if global_step % 50 == 0:
                        memory_weight_now = warmup_schedule.get_memory_weight(global_step)  # now LTM-only
                        phase_name_now = warmup_schedule.get_phase_name(global_step)
                        aux_mult_now = warmup_schedule.get_aux_weight_multiplier(global_step)

                        metrics["train_steps"].append(global_step)
                        # Standard LM metrics
                        metrics["train_loss"].append(
                            (running_main_loss_sum / running_tokens) if running_tokens > 0 else 0.0
                        )
                        metrics["train_acc"].append(
                            (running_correct / running_tokens) if running_tokens > 0 else 0.0
                        )
                        metrics.setdefault("train_loss_no_eos", []).append(
                            (running_main_loss_sum_no_eos / running_tokens_no_eos) if running_tokens_no_eos > 0 else 0.0
                        )
                        # Per-step component means
                        metrics["main_loss_per_step"].append(
                            (running_main_step_loss_sum / running_main_step_count) if running_main_step_count > 0 else 0.0
                        )
                        metrics.setdefault("main_loss_per_step_no_eos", []).append(
                            (running_main_step_loss_sum_no_eos / running_main_step_count_no_eos) if running_main_step_count_no_eos > 0 else 0.0
                        )
                        metrics["bow_loss_per_step"].append(
                            running_bow_step_sum / max(1, running_bow_step_count)
                        )
                        metrics["adjacency_loss_per_step"].append(
                            running_adj_step_sum / max(1, running_adj_step_count)
                        )
                        metrics["alignment_loss_per_step"].append(
                            running_align_step_sum / max(1, running_align_step_count)
                        )
                        metrics["alignment_accuracy"].append(
                            (running_align_acc / running_align_count) if running_align_count > 0 else 0.0
                        )
                        metrics["norm_reg_loss_per_step"].append(
                            running_normreg_step_sum / max(1, running_normreg_step_count)
                        )
                        metrics["gate_reg_loss_per_step"].append(
                            running_gate_reg_step_sum / max(1, running_gate_reg_step_count)
                        )

                        metrics.setdefault("bow_accuracy", []).append(
                            (running_bow_acc / running_bow_count) if running_bow_count > 0 else 0.0
                        )
                        metrics.setdefault("adjacency_accuracy", []).append(
                            (running_adj_acc / running_adj_count) if running_adj_count > 0 else 0.0
                        )

                        # Train-side TOTAL loss per step (sum of component means) and derived perplexity
                        main_token_avg = (running_main_loss_sum / running_tokens) if running_tokens > 0 else 0.0
                        metrics.setdefault("train_ppl_main_token", []).append(_to_ppl(main_token_avg))
                        main_token_avg_no_eos = (
                            running_main_loss_sum_no_eos / running_tokens_no_eos
                        ) if running_tokens_no_eos > 0 else 0.0
                        metrics.setdefault("train_ppl_no_eos", []).append(_to_ppl(main_token_avg_no_eos))

                        metrics.setdefault("train_loss_sum", []).append(running_main_loss_sum)
                        metrics.setdefault("train_loss_sum_no_eos", []).append(running_main_loss_sum_no_eos)
                        metrics.setdefault("train_eos_loss_sum", []).append(running_eos_loss_sum)
                        metrics.setdefault("train_token_count", []).append(int(running_tokens))
                        metrics.setdefault("train_token_count_no_eos", []).append(int(running_tokens_no_eos))
                        metrics.setdefault("train_eos_count", []).append(int(running_eos_count))

                        # Aux info
                        metrics["lr"].append(sched.get_last_lr()[0])
                        metrics["ltm_size"].append(ltm.size() if ltm else 0)
                        metrics["memory_weight"].append(memory_weight_now)
                        metrics["phase"].append(phase_name_now)
                        metrics["aux_weight_mult"].append(aux_mult_now)

                        if actual_stm_sizes_batch:
                            metrics["actual_stm_sizes"].extend(actual_stm_sizes_batch)
                            actual_stm_sizes_batch = []

                        # Per-layer gates: log both memory and self-attn gates
                        for i_blk, block in enumerate(model.blocks):
                            metrics.setdefault(f"gate_{i_blk}", []).append(float(block.memory_gate.item()))
                            metrics.setdefault(f"self_gate_{i_blk}", []).append(float(block.self_gate.item()))

                        # S_REP analytics
                        if srep_embs_all:
                            srep_embs_all_cat = torch.cat([s.detach() for s in srep_embs_all], dim=0).to(device)
                            update_metrics_with_srep_analytics(
                                metrics, srep_diagnostics,
                                srep_embs_all_cat, srep_texts_all, srep_doc_ids_all, srep_raw_norms_all
                            )

                        ltm_had_values = False
                        if batch_ltm_stats:
                            update_metrics_with_ltm_stats(metrics, batch_ltm_stats)
                            ltm_had_values = True
                        if batch_query_comparison:
                            update_metrics_with_query_comparison(metrics, batch_query_comparison)
                            ltm_had_values = True
                        if ltm_had_values:
                            metrics.setdefault("ltm_steps", []).append(global_step)

                        # GPU memory & speed
                        if torch.cuda.is_available():
                            metrics["gpu_memory_gb"].append(torch.cuda.max_memory_allocated() / 1e9)
                        elapsed = time.time() - t0
                        metrics["steps_per_second"].append(global_step / max(1e-9, elapsed))

                    # -------- Evaluation cadence --------
                    if global_step % args.eval_every == 0:
                        memory_weight_eval = warmup_schedule.get_memory_weight(global_step)
                        aux_mult_now = warmup_schedule.get_aux_weight_multiplier(global_step)

                        val_loss, val_acc, val_attn_metrics, val_eval = evaluate(
                            model, ltm, val_loader, device, memory_weight_eval,
                            aux_losses=aux_losses, args=args, aux_weight_mult=aux_mult_now,
                            debug_dump_dir=None,                # disable old LTM retrieval debug
                            debug_dump_n_sentences=5,
                            debug_dump_n_docs=5,
                            debug_dump_topk=5,
                            global_step=global_step
                        )

                        # ---- Debug inference rollout (replaces LTM retrieval logs) ----
                        try:
                            latest_inf = os.path.join(em.path("debug"), "val_inference_latest.txt")
                            rollout_from_loader(
                                model,
                                tokenizer,
                                val_loader,
                                latest_inf,
                                max_tokens=200,
                                max_sentences=10,
                                seed=random.randint(0, 10**9),
                            )
                            logger.info(f"Validation inference rollout written to: {latest_inf}")
                        except Exception as e:
                            logger.warning(f"Inference rollout during eval failed: {e}")


                        # Record standard LM metrics
                        metrics["val_steps"].append(global_step)
                        metrics["val_loss"].append(val_loss)
                        metrics.setdefault("val_loss_no_eos", []).append(
                            float(val_eval.get("losses", {}).get("main_token_no_eos", val_loss))
                        )
                        metrics["val_acc"].append(val_acc)
                        for key, value in val_attn_metrics.items():
                            metrics.setdefault(f"val_{key}", []).append(value)

                        metrics.setdefault("val_ppl_main_token", []).append(
                            float(val_eval.get("perplexities", {}).get("main_token", _to_ppl(val_loss)))
                        )
                        metrics.setdefault("val_ppl_no_eos", []).append(
                            float(val_eval.get("perplexities", {}).get("main_token_no_eos", _to_ppl(val_loss)))
                        )
                        metrics.setdefault("val_loss_sum", []).append(
                            float(val_eval.get("loss_sums", {}).get("main_token", 0.0))
                        )
                        metrics.setdefault("val_loss_sum_no_eos", []).append(
                            float(val_eval.get("loss_sums", {}).get("main_token_no_eos", 0.0))
                        )
                        metrics.setdefault("val_eos_loss_sum", []).append(
                            float(val_eval.get("loss_sums", {}).get("main_token_eos", 0.0))
                        )
                        metrics.setdefault("val_token_count", []).append(
                            int(val_eval.get("token_counts", {}).get("total", 0))
                        )
                        metrics.setdefault("val_token_count_no_eos", []).append(
                            int(val_eval.get("token_counts", {}).get("no_eos", 0))
                        )
                        metrics.setdefault("val_eos_count", []).append(
                            int(val_eval.get("token_counts", {}).get("eos", 0))
                        )

                        # Human-friendly logging
                        train_lm_ppl = metrics["train_ppl_main_token"][-1] if metrics.get("train_ppl_main_token") else float('nan')
                        val_lm_ppl   = metrics["val_ppl_main_token"][-1]   if metrics.get("val_ppl_main_token")   else float('nan')
                        logger.info(
                            f"[Step {global_step}] Phase={warmup_schedule.get_phase_name(global_step)} "
                            f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}  "
                            f"Val LM PPL={val_lm_ppl:.3f}  "
                            f"Train LM PPL={train_lm_ppl:.3f}"
                        )

                        if not dropout_enabled and args.dropout_activation_delta > 0:
                            if val_lm_ppl == val_lm_ppl:  # NaN safe check
                                if prev_val_perplexity is not None:
                                    improvement = prev_val_perplexity - val_lm_ppl
                                    if improvement < args.dropout_activation_delta:
                                        dropout_enabled = True
                                        dropout_activation_step = int(global_step)
                                        logger.info(
                                            f"Activating scheduled context dropout at step {global_step} "
                                            f"(Δppl={improvement:.3f} < {args.dropout_activation_delta})."
                                        )
                                prev_val_perplexity = val_lm_ppl

                        val_loss_no_eos = float(val_eval.get("losses", {}).get("main_token_no_eos", val_loss))

                        # Save ONLY the best checkpoint (same policy as before)
                        # ---- Save ONLY the best checkpoint (best-only policy) ----
                        if val_loss_no_eos < best_val_loss_no_eos:
                            best_val_loss_no_eos = float(val_loss_no_eos)
                            ckpt_dir = os.path.join(em.path("checkpoints"), "best")
                            os.makedirs(ckpt_dir, exist_ok=True)

                            # (1) Model weights
                            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))

                            # (2) Trainer state: optimizer, LR scheduler, AMP scaler
                            torch.save({
                                "optimizer": optim.state_dict(),
                                "scheduler": sched.state_dict(),
                                "scaler": scaler.state_dict(),
                            }, os.path.join(ckpt_dir, "train_state.pt"))

                            # (3) Long-term memory (only in global LTM mode)
                            meta = {
                                "global_step": int(global_step),
                                "best_val_loss_no_eos": float(best_val_loss_no_eos),
                                "best_val_loss": float(best_val_loss_no_eos),
                                "dropout_enabled": dropout_enabled,
                                "prev_val_perplexity": prev_val_perplexity,
                                "dropout_activation_step": dropout_activation_step,
                            }
                            if isinstance(ltm, LongTermMemoryGPU):
                                torch.save(
                                    {
                                        "texts": ltm.texts,
                                        "sentence_to_idx": ltm.sentence_to_idx,
                                        "embeddings": ltm.embeddings.detach().cpu(),
                                    },
                                    os.path.join(ckpt_dir, "ltm.pt"),
                                )
                                meta["ltm_file"] = "ltm.pt"

                            # (4) Metadata (step, best loss, etc.)
                            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                                json.dump(meta, f, indent=2)

                            logger.info(f"✓ Saved new best checkpoint → {ckpt_dir}")


                        model.train()


                    # Advance global step at each sentence-step
                    global_step += 1

                # -------- backward/step (AMP-aware), once per (accumulated) batch --------
                if step_losses:
                    total_batch_loss = torch.stack(step_losses).mean()
                    scaled = scaler.scale(total_batch_loss / args.gradient_accumulation_steps)
                    scaled.backward()

                if step_losses:
                    # Returns norm of *scaled* grads; divide by AMP scale to get true norm
                    total_norm_scaled = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
                    scale_val = float(scaler.get_scale()) if amp_enabled else 1.0
                    if scale_val <= 0.0:
                        scale_val = 1.0
                    total_norm_true = float(total_norm_scaled.item()) / scale_val
                    metrics.setdefault("grad_norm", []).append(total_norm_true)
                    metrics.setdefault("grad_norm_steps", []).append(global_step)

                # Current AMP scale (for unscaling probe readings)
                scale = float(scaler.get_scale()) if amp_enabled else 1.0
                unscale = (scale if scale > 0.0 else 1.0)

                # STM S_REP grad probe (AMP-aware)
                for step_idx, mean_norm in probe.measure_and_reset():
                    mean_norm /= unscale
                    metrics.setdefault("stm_srep_grad_steps", []).append(int(step_idx))
                    metrics.setdefault("stm_srep_grad", []).append(float(mean_norm))


                # -------- Parameter-level component grad norms --------
                # Only sample at your cadence to keep overhead negligible
                if will_probe_comp:
                    avg, blk0, blklast = collect_param_component_grad_norms(model)
                    step_idx = int(comp_probe_step)
                    # All blocks (average)
                    for k, v in avg.items():
                        metrics.setdefault(f"comp_grad_{k}_steps", []).append(step_idx)
                        metrics.setdefault(f"comp_grad_{k}", []).append(float(v / unscale))

                    # First block (L0)
                    for k, v in blk0.items():
                        metrics.setdefault(f"comp_grad_{k}_L0_steps", []).append(step_idx)
                        metrics.setdefault(f"comp_grad_{k}_L0", []).append(float(v / unscale))

                    # Last block (L{n-1})
                    last_idx = model.cfg.n_layers - 1
                    for k, v in blklast.items():
                        metrics.setdefault(f"comp_grad_{k}_L{last_idx}_steps", []).append(step_idx)
                        metrics.setdefault(f"comp_grad_{k}_L{last_idx}", []).append(float(v / unscale))


                # Optimizer step (grad clip, step, update, zero)
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Unscale once per real optimizer step, then clip and step
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                    sched.step()
                    optim.zero_grad(set_to_none=True)

                # --- draw AFTER probes have written their metrics ---
                if (global_step == 0) or (args.plot_every > 0 and  _batch_will_hit_cadence(global_step, batch_docs, args.plot_every)):
                    print(f"Plotting at step {global_step} (batch {batch_idx} epoch {epoch})")
                    plotter.plot_all(metrics, global_step, epoch_window=args.epoch_window)

                # Update LTM after batch (store detached normalized S_REPs)
                if ltm is not None and srep_embs_all:
                    srep_embs_cat = torch.cat([s.detach() for s in srep_embs_all], dim=0).to(device)
                    if isinstance(ltm, LongTermMemoryGPU):
                        adds, updates, repl_sims = ltm.add_or_update_batch(srep_texts_all, srep_embs_cat)
                    else:
                        # Document-scoped: route by original_doc_id
                        adds, updates, repl_sims = ltm.add_or_update_batch(srep_texts_all, srep_embs_cat, srep_doc_ids_all)
                    if repl_sims:
                        import numpy as _np
                        mean_repl = float(_np.mean(repl_sims))
                        metrics.setdefault("ltm_replace_sim", []).append(mean_repl)        # keep existing plot key
                        metrics.setdefault("ltm_replace_steps", []).append(global_step)

        plotter.plot_all(metrics, global_step, epoch_window=args.epoch_window)
        plotter.snapshot(epoch)

        # Save a stable epoch snapshot of the most recent validation debug file
        latest_inf = os.path.join(em.path("debug"), "val_inference_latest.txt")
        if os.path.exists(latest_inf):
            epoch_inf = os.path.join(em.path("debug"), f"val_inference_epoch_{epoch+1}.txt")
            try:
                shutil.copy2(latest_inf, epoch_inf)
                logger.info(f"Copied latest inference debug to epoch snapshot: {epoch_inf}")
            except Exception as e:
                logger.warning(f"Failed to copy epoch inference debug file: {e}")


        # End of epoch: DO NOT save epoch checkpoints (best only policy)


    # -----------------------------
    # Final evaluation + save metrics & plots
    # -----------------------------
    aux_mult_now = warmup_schedule.get_aux_weight_multiplier(global_step)
    val_loss, val_acc, val_attn_metrics, val_eval = evaluate(
        model,
        ltm,
        val_loader,
        device,
        1.0,
        aux_losses=aux_losses,
        args=args,
        aux_weight_mult=aux_mult_now,
        debug_dump_dir=None,
        debug_dump_n_sentences=5,
        debug_dump_n_docs=5,
        debug_dump_topk=5,
        global_step=global_step,
    )
    metrics["val_steps"].append(global_step)
    metrics["val_loss"].append(val_loss)
    metrics.setdefault("val_loss_no_eos", []).append(
        float(val_eval.get("losses", {}).get("main_token_no_eos", val_loss))
    )
    metrics["val_acc"].append(val_acc)
    for key, value in val_attn_metrics.items():
        metrics.setdefault(f"val_{key}", []).append(value)

    metrics.setdefault("val_ppl_main_token", []).append(float(val_eval.get("perplexities", {}).get("main_token", _to_ppl(val_loss))))
    metrics.setdefault("val_ppl_no_eos", []).append(
        float(val_eval.get("perplexities", {}).get("main_token_no_eos", _to_ppl(val_loss)))
    )
    metrics.setdefault("val_loss_sum", []).append(float(val_eval.get("loss_sums", {}).get("main_token", 0.0)))
    metrics.setdefault("val_loss_sum_no_eos", []).append(float(val_eval.get("loss_sums", {}).get("main_token_no_eos", 0.0)))
    metrics.setdefault("val_eos_loss_sum", []).append(float(val_eval.get("loss_sums", {}).get("main_token_eos", 0.0)))
    metrics.setdefault("val_token_count", []).append(int(val_eval.get("token_counts", {}).get("total", 0)))
    metrics.setdefault("val_token_count_no_eos", []).append(int(val_eval.get("token_counts", {}).get("no_eos", 0)))
    metrics.setdefault("val_eos_count", []).append(int(val_eval.get("token_counts", {}).get("eos", 0)))

    plotter.plot_all(metrics, global_step, epoch_window=args.epoch_window)
    # Final snapshot of the latest retrieval debug
    try:
        latest_inf = os.path.join(em.path("debug"), "val_inference_latest.txt")
        rollout_from_loader(
            model,
            tokenizer,
            val_loader,
            latest_inf,
            max_tokens=200,
            max_sentences=10,
            seed=random.randint(0, 10**9),
        )
        logger.info(f"Final validation inference rollout written to: {latest_inf}")
    except Exception as e:
        logger.warning(f"Final inference rollout failed: {e}")

    final_inf = os.path.join(em.path("debug"), "val_inference_latest.txt")
    if os.path.exists(final_inf):
        final_epoch_inf = os.path.join(em.path("debug"), "val_inference_epoch_final.txt")
        try:
            logger.info(f"Saved final inference debug snapshot: {final_epoch_inf}")
        except Exception as e:
            logger.warning(f"Failed to save final inference debug snapshot: {e}")

    with open(os.path.join(em.path("logs"), "metrics.json"), "w") as f:
        json.dump({k: list(v) if isinstance(v, (list, deque)) else v for k, v in metrics.items()}, f, indent=2)

    mins = (time.time() - t0) / 60.0
    logger.info(f"Training complete in {mins:.1f} min. Best val loss (no EOS): {best_val_loss_no_eos:.4f}")


if __name__ == "__main__":
    main()
