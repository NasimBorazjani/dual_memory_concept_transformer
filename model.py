#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Sentence-level Transformer with all requested changes:
- Attention capture fixed (average_attn_weights=False)
- NO residual alpha - only memory gate
- STM positional encoding: L2-scaled, keys-only
- Adjacency pairs generation for loss
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Deque, Any
from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Configuration
# -----------------------------
from dataclasses import dataclass

@dataclass
class SentenceTransformerConfig:
    # Core transformer
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    ffw_mult: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # Sentence / memory representation space (S_REP latent == memory K/V)
    srep_dim: int = 1024                 # single knob for S_REP latent + memory bank K/V

    # Sequence / vocab
    max_position_embeddings: int = 66    # 64 tokens + [EOS] + [S_REP]
    vocab_size: int = 50257
    max_sentence_tokens: int = 64        # excluding [EOS],[S_REP]

    # Memory settings
    max_sentences_in_short_term: int = 15
    use_long_term_memory: bool = True
    no_ltm_for_first_k_blocks: int = 4
    ltm_top_k: int = 15
    ltm_min_sim: float = 0.1
    ltm_min_matches: int = 2
    ltm_query_mode: str = "tokens_only"  # "tokens_only", "hybrid", "both"

    # Optional STM positional encoding (applied to STM KEYS only; L2-scaled)
    use_stm_positional: bool = False
    stm_positional_weight: float = 1.0

    # Gates / dropout
    memory_gate_init: float = 1.0
    context_dropout: float = 0.1         # scheduled at runtime; this is the max/reference value
    srep_dropout: float = 0.1

    # S_REP norm control
    srep_norm_target: float = 1.0
    srep_norm_margin: float = 0.1
    srep_norm_reg_weight: float = 0.001

    # Debug / toggles
    debug_no_memory: bool = False
    debug_stm_only: bool = False

    # ---- S_REP head / pooling knobs ----
    srep_head_depth: int = 2             # 1 = linear, 2 = MLP, 3 = deeper MLP
    srep_head_mult: int = 4              # expansion factor in head hidden
    srep_head_activation: str = "gelu"   # "gelu" or "swiglu"

    use_attentive_pool: bool = False     # if True, add learned-query pooling over visible tokens
    attn_pool_n_queries: int = 1         # number of learned queries (in d_model space)


# -----------------------------
# STM Positional Encoding (L2-scaled)
# -----------------------------
class STMPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding scaled by L2 norm; dimension = srep_dim."""
    def __init__(self, dim: int, weight_l2: float = 0.1):
        super().__init__()
        self.dim = dim
        self.weight_l2 = weight_l2

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        device = positions.device
        D = self.dim
        pe = torch.zeros(len(positions), D, device=device)
        div_term = torch.exp(torch.arange(0, D, 2, device=device) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
        pe = F.normalize(pe, p=2, dim=1) * self.weight_l2
        return pe

# -----------------------------
# Enhanced GPU-based LTM with multiple query modes
# -----------------------------
class LongTermMemoryGPU:
    """GPU-optimized long-term memory with flexible retrieval modes."""
    def __init__(self, embedding_dim: int, device: torch.device):
        self.embedding_dim = embedding_dim
        self.device = device
        self.embeddings = torch.empty(0, embedding_dim, device=device)  # [N, D]
        self.texts: List[str] = []
        self.sentence_to_idx: Dict[str, int] = {}

    def size(self) -> int:
        return len(self.texts)

    def add_or_update_batch(self, sentences: List[str], embeddings: torch.Tensor) -> Tuple[int, int, List[float]]:
        """Add or update sentence embeddings (detached + L2-normalized).
        Returns (adds, updates, update_sims) where update_sims are cosine similarities between
        old and new embeddings only for updated entries.
        """
        if embeddings.numel() == 0 or not sentences:
            return 0, 0, []

        with torch.no_grad():
            embs = F.normalize(embeddings.detach().to(self.device), p=2, dim=1)

        adds, updates = 0, 0
        update_sims: List[float] = []

        for sent, emb in zip(sentences, embs):
            if sent in self.sentence_to_idx:
                idx = self.sentence_to_idx[sent]
                # both old and emb are unit-norm
                sim = torch.dot(self.embeddings[idx], emb).item()
                update_sims.append(sim)
                self.embeddings[idx] = emb
                updates += 1
            else:
                self.sentence_to_idx[sent] = len(self.texts)
                self.texts.append(sent)
                self.embeddings = torch.cat([self.embeddings, emb.unsqueeze(0)], dim=0)
                adds += 1

        return adds, updates, update_sims

    @torch.no_grad()
    def retrieve_gpu(
        self,
        query_embeddings: torch.Tensor,    # [B, D]
        top_k: int = 15,
        min_sim: float = 0.1,
        min_matches: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], Dict]:
        """
        Returns:
          - out_embs: [B, K_used, D] padded with zeros where masked
          - out_mask: [B, K_used] True where padded/invalid
          - out_texts: List of retrieved sentences (per batch row)
          - stats: dict with retrieval statistics + detailed per-row items:
              {
                'max_sim','mean_sim','used_ratio','first_sim','mean_topk','mean_valid',
                'items_by_row': [[{'text','sim','rank','initial_valid'}, ...], ...],
                'used_row_flags': [bool, ...]
              }
        """
        B, D = query_embeddings.shape
        stats = {
            'max_sim': 0.0, 'mean_sim': 0.0, 'used_ratio': 0.0,
            'first_sim': 0.0, 'mean_topk': 0.0, 'mean_valid': 0.0,
            'items_by_row': [],
            'used_row_flags': []
        }

        if self.size() == 0:
            empty = query_embeddings.new_zeros((B, 0, D))
            empty_mask = torch.ones((B, 0), dtype=torch.bool, device=self.device)
            stats['items_by_row'] = [[] for _ in range(B)]
            stats['used_row_flags'] = [False for _ in range(B)]
            return empty, empty_mask, [[] for _ in range(B)], stats

        Q = F.normalize(query_embeddings.to(self.device), p=2, dim=1)
        sims = torch.mm(Q, self.embeddings.t())  # [B, N]
        k_actual = min(top_k, self.size())

        top_sims, top_idx = sims.topk(k_actual, dim=1) if k_actual > 0 else (None, None)
        if k_actual <= 0:
            empty = query_embeddings.new_zeros((B, 0, D))
            empty_mask = torch.ones((B, 0), dtype=torch.bool, device=self.device)
            stats['items_by_row'] = [[] for _ in range(B)]
            stats['used_row_flags'] = [False for _ in range(B)]
            return empty, empty_mask, [[] for _ in range(B)], stats

        valid_mask = top_sims >= min_sim                # [B, k]
        valid_counts = valid_mask.sum(dim=1)            # [B]
        use_ltm = valid_counts >= min_matches           # [B]

        # aggregate stats
        stats['max_sim'] = top_sims[:, 0].mean().item()
        stats['first_sim'] = stats['max_sim']
        stats['mean_topk'] = top_sims.mean().item()
        if valid_mask.any():
            stats['mean_sim'] = top_sims[valid_mask].mean().item()
            stats['mean_valid'] = stats['mean_sim']
        stats['used_ratio'] = use_ltm.float().mean().item()

        # detailed per-row items
        items_by_row: List[List[Dict[str, Any]]] = []
        for b in range(B):
            row_items: List[Dict[str, Any]] = []
            for j in range(k_actual):
                idx = int(top_idx[b, j].item())
                sim = float(top_sims[b, j].item())
                text = self.texts[idx]
                row_items.append({
                    'text': text,
                    'sim': sim,
                    'rank': j + 1,
                    'initial_valid': bool(valid_mask[b, j].item())
                })
            items_by_row.append(row_items)
        stats['items_by_row'] = items_by_row
        stats['used_row_flags'] = [bool(x.item()) for x in use_ltm]

        # build outputs containing only valid entries if row is usable
        max_valid = int(valid_mask.sum(dim=1).max().item()) if valid_mask.any() else 0
        out_embs = query_embeddings.new_zeros((B, max_valid, D))
        out_mask = torch.ones((B, max_valid), dtype=torch.bool, device=self.device)
        out_texts: List[List[str]] = []

        for b in range(B):
            if use_ltm[b] and max_valid > 0:
                valid_idx = top_idx[b][valid_mask[b]]
                n_valid = valid_idx.size(0)
                if n_valid > 0:
                    out_embs[b, :n_valid] = self.embeddings[valid_idx]
                    out_mask[b, :n_valid] = False
                    out_texts.append([self.texts[i.item()] for i in valid_idx])
                else:
                    out_texts.append([])
            else:
                out_texts.append([])

        return out_embs, out_mask, out_texts, stats


# -----------------------------
# Document-scoped LTM (per-document stores)
# -----------------------------
class DocumentLongTermMemory:
    """
    Holds a separate LongTermMemoryGPU for each document (keyed by original_doc_id).
    Retrieval and updates are routed per-row by doc_id. Stats are aggregated across rows.
    """
    def __init__(self, embedding_dim: int, device: torch.device):
        self.embedding_dim = int(embedding_dim)
        self.device = device
        self.by_doc: Dict[str, LongTermMemoryGPU] = {}

    def reset_document(self, doc_id: str) -> None:
        """Clear the per-document store at a doc boundary."""
        self.by_doc.pop(str(doc_id), None)

    def reset_all(self) -> None:
        """Clear all per-document stores (rarely needed)."""
        self.by_doc.clear()

    def _get_mem(self, doc_id: str) -> LongTermMemoryGPU:
        if doc_id not in self.by_doc:
            self.by_doc[doc_id] = LongTermMemoryGPU(self.embedding_dim, self.device)
        return self.by_doc[doc_id]

    def size(self) -> int:
        return sum(mem.size() for mem in self.by_doc.values())

    @torch.no_grad()
    def retrieve(
        self,
        query_embeddings: torch.Tensor,        # [B, D]
        doc_ids: List[str],                    # len B
        top_k: int = 15,
        min_sim: float = 0.1,
        min_matches: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], Dict]:
        """
        Returns:
          - out_embs: [B, Kmax, D] (zero-padded)
          - out_mask: [B, Kmax]    (True where padded/invalid)
          - out_texts: list of retrieved sentence texts per row
          - stats: dict (aggregated) + detailed fields:
              {
                'max_sim','first_sim','mean_topk','mean_valid','used_ratio',
                'items_by_row': [[{'text','sim','rank','initial_valid'}, ...], ...],
                'used_row_flags': [bool, ...]
              }
        """
        B, D = query_embeddings.shape
        if B == 0:
            empty = query_embeddings.new_zeros((0, 0, D))
            empty_mask = torch.ones((0, 0), dtype=torch.bool, device=query_embeddings.device)
            return empty, empty_mask, [], {
                'max_sim': 0.0, 'first_sim': 0.0, 'mean_topk': 0.0, 'mean_valid': 0.0, 'used_ratio': 0.0,
                'items_by_row': [], 'used_row_flags': []
            }

        Q = F.normalize(query_embeddings.detach().to(self.device), p=2, dim=1)

        per_row_first, per_row_mean_topk, per_row_mean_valid, per_row_used = [], [], [], []
        row_results: List[Tuple[torch.Tensor, torch.Tensor, List[str]]] = []
        items_by_row: List[List[Dict[str, Any]]] = []
        used_row_flags: List[bool] = []

        for b in range(B):
            doc_id = doc_ids[b]
            mem = self._get_mem(doc_id)
            if mem.size() == 0:
                row_results.append((Q.new_zeros((0, D)), torch.ones(0, dtype=torch.bool, device=self.device), []))
                per_row_first.append(0.0); per_row_mean_topk.append(0.0); per_row_mean_valid.append(0.0); per_row_used.append(0.0)
                items_by_row.append([]); used_row_flags.append(False)
                continue

            sims = torch.mv(mem.embeddings, Q[b])             # [N]
            k_actual = min(top_k, mem.size())
            if k_actual > 0:
                top_sims, top_idx = sims.topk(k_actual)
                valid_mask = top_sims >= min_sim
                valid_count = int(valid_mask.sum().item())
                use_ltm = (valid_count >= min_matches)

                per_row_first.append(float(top_sims[0].item()))
                per_row_mean_topk.append(float(top_sims.mean().item()))
                per_row_mean_valid.append(float(top_sims[valid_mask].mean().item()) if valid_count > 0 else 0.0)
                per_row_used.append(1.0 if use_ltm else 0.0)
                used_row_flags.append(bool(use_ltm))

                # detailed items in rank order
                row_items: List[Dict[str, Any]] = []
                for j in range(k_actual):
                    idx = int(top_idx[j].item())
                    row_items.append({
                        'text': mem.texts[idx],
                        'sim': float(top_sims[j].item()),
                        'rank': j + 1,
                        'initial_valid': bool(valid_mask[j].item())
                    })
                items_by_row.append(row_items)

                if use_ltm and valid_count > 0:
                    used_idx = top_idx[valid_mask]            # [valid_count]
                    embs = mem.embeddings[used_idx]            # [valid_count, D]
                    texts = [mem.texts[i.item()] for i in used_idx]
                    mask = torch.zeros(valid_count, dtype=torch.bool, device=self.device)
                    row_results.append((embs, mask, texts))
                else:
                    row_results.append((Q.new_zeros((0, D)), torch.ones(0, dtype=torch.bool, device=self.device), []))
            else:
                per_row_first.append(0.0); per_row_mean_topk.append(0.0); per_row_mean_valid.append(0.0); per_row_used.append(0.0)
                used_row_flags.append(False)
                items_by_row.append([])
                row_results.append((Q.new_zeros((0, D)), torch.ones(0, dtype=torch.bool, device=self.device), []))

        # Collate to rectangular [B, Kmax, D]
        Kmax = max((x[0].size(0) for x in row_results), default=0)
        out_embs = Q.new_zeros((B, Kmax, D))
        out_mask = torch.ones((B, Kmax), dtype=torch.bool, device=self.device)
        out_texts: List[List[str]] = []
        for b, (embs_b, mask_b, texts_b) in enumerate(row_results):
            k_b = embs_b.size(0)
            if k_b > 0:
                out_embs[b, :k_b] = embs_b
                out_mask[b, :k_b] = mask_b
            out_texts.append(texts_b)

        stats = {
            'max_sim': float(torch.tensor(per_row_first).mean().item()) if per_row_first else 0.0,
            'first_sim': float(torch.tensor(per_row_first).mean().item()) if per_row_first else 0.0,
            'mean_topk': float(torch.tensor(per_row_mean_topk).mean().item()) if per_row_mean_topk else 0.0,
            'mean_valid': float(torch.tensor(per_row_mean_valid).mean().item()) if per_row_mean_valid else 0.0,
            'used_ratio': float(torch.tensor(per_row_used).mean().item()) if per_row_used else 0.0,
            'items_by_row': items_by_row,
            'used_row_flags': used_row_flags,
        }
        return out_embs, out_mask, out_texts, stats

    def add_or_update_batch(
        self,
        sentences: List[str],
        embeddings: torch.Tensor,      # [N, D]
        doc_ids: List[str]
    ) -> Tuple[int, int, List[float]]:
        """
        Route (sent, emb) to the right per-document store. Returns (adds, updates, update_sims[]).
        """
        if embeddings.numel() == 0 or not sentences:
            return 0, 0, []
        adds = updates = 0
        sims_all: List[float] = []

        # group by doc to batch calls
        from collections import defaultdict
        groups = defaultdict(list)
        for s, e, d in zip(sentences, embeddings, doc_ids):
            groups[d].append((s, e))

        for d, pairs in groups.items():
            mem = self._get_mem(d)
            sents = [p[0] for p in pairs]
            embs  = torch.stack([p[1] for p in pairs], dim=0).to(self.device)
            a, u, sims = mem.add_or_update_batch(sents, embs)
            adds += a; updates += u
            sims_all.extend(sims)

        return adds, updates, sims_all


# -----------------------------
# Transformer block (Pre-LN, MEMORY FIRST) with linear gates
# -----------------------------
class SentenceBlock(nn.Module):
    def __init__(self, cfg: SentenceTransformerConfig):
        super().__init__()
        d = cfg.d_model

        # Pre-LN layers
        self.ln_mem  = nn.LayerNorm(d)
        self.ln_self = nn.LayerNorm(d)
        self.ln_ffn  = nn.LayerNorm(d)

        # Cross- and self-attention  (K/V live in srep_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.n_heads, dropout=cfg.attn_dropout, batch_first=True,
            kdim=cfg.srep_dim, vdim=cfg.srep_dim
        )
        self.self_attn  = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.n_heads, dropout=cfg.attn_dropout, batch_first=True
        )

        # Gates (trainable scalars)
        self.memory_gate = nn.Parameter(torch.tensor(cfg.memory_gate_init))
        if cfg.debug_no_memory:
            for p in self.cross_attn.parameters():
                p.requires_grad = False
            self.memory_gate.requires_grad = False
            self.self_gate = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        else:
            self.self_gate = nn.Parameter(torch.tensor(1.0))

        # ---- Freeze both gates at 1.0 (constant identity; no gradients) ----
        self.memory_gate.data.fill_(1.0)
        self.memory_gate.requires_grad = False
        self.self_gate.data.fill_(1.0)
        self.self_gate.requires_grad = False

        self.drop = nn.Dropout(cfg.dropout)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.ffw_mult * d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffw_mult * d, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_pad_mask: torch.Tensor,
        k_memory_keys: Optional[torch.Tensor],
        k_memory_vals: Optional[torch.Tensor],
        k_memory_mask: Optional[torch.Tensor],
        use_memory: bool = True,
        memory_weight: float = 1.0,
        srep_positions: Optional[torch.Tensor] = None,
        return_srep_deltas: bool = False,
    ):
        mem_inc = None

        if not use_memory:
            q_self = self.ln_self(x)
            s_out, _ = self.self_attn(
                q_self, q_self, q_self,
                attn_mask=attn_mask, key_padding_mask=key_pad_mask,
                need_weights=False,
                is_causal=True,
            )
            self_inc = self.drop(self.self_gate * s_out)
            x = x + self_inc

            q_ffn = self.ln_ffn(x)
            ffn_inc = self.mlp(q_ffn)
            x = x + ffn_inc

            if return_srep_deltas:
                comp_full = {'cross_attn': None, 'self_attn': self_inc, 'ffn': ffn_inc, 'layernorm': q_self}
                return x, comp_full
            return x

        # With memory
        if k_memory_keys is not None and k_memory_keys.size(1) > 0:
            q_mem = self.ln_mem(x)
            m_out, _ = self.cross_attn(
                q_mem, k_memory_keys, k_memory_vals,
                key_padding_mask=k_memory_mask,
                need_weights=False, average_attn_weights=False
            )
            mem_inc = self.drop(self.memory_gate * memory_weight * m_out)
            x = x + mem_inc

        q_self = self.ln_self(x)
        s_out, _ = self.self_attn(
            q_self, q_self, q_self,
            attn_mask=attn_mask, key_padding_mask=key_pad_mask,
            need_weights=False
        )
        self_inc = self.drop(self.self_gate * s_out)
        x = x + self_inc

        q_ffn = self.ln_ffn(x)
        ffn_inc = self.mlp(q_ffn)
        x = x + ffn_inc

        if return_srep_deltas:
            comp_full = {'cross_attn': mem_inc, 'self_attn': self_inc, 'ffn': ffn_inc, 'layernorm': q_self}
            return x, comp_full
        return x


# -----------------------------
# Utilities for the S_REP head
# -----------------------------
class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

def _make_srep_head(cfg: SentenceTransformerConfig) -> nn.Module:
    """
    SimCLR-style MLP head.
    Depth=1 -> Linear(d_model -> srep_dim)
    Depth>=2 -> [Linear/Gated -> Act -> Dropout] x (depth-1) -> Linear(hidden -> srep_dim)
    Hidden width = cfg.srep_head_mult * cfg.srep_dim
    """
    d_in  = cfg.d_model
    d_out = cfg.srep_dim
    depth = max(1, int(cfg.srep_head_depth))
    hidden = max(d_out, int(cfg.srep_head_mult) * d_out)

    if depth == 1:
        return nn.Linear(d_in, d_out)

    layers: List[nn.Module] = []

    def block(in_dim: int, out_dim: int):
        if cfg.srep_head_activation.lower() == "swiglu":
            layers.append(nn.Linear(in_dim, 2 * out_dim))
            layers.append(SwiGLU())
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
        layers.append(nn.Dropout(cfg.dropout))

    # first (d_in -> hidden)
    block(d_in, hidden)

    # optional middle blocks (hidden -> hidden)
    for _ in range(depth - 2):
        block(hidden, hidden)

    # final projection to d_out
    layers.append(nn.Linear(hidden, d_out))
    return nn.Sequential(*layers)



# -----------------------------
# Main model with enhanced features
# -----------------------------
class SentenceTransformer(nn.Module):
    def __init__(self, cfg: SentenceTransformerConfig, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        # Special token IDs
        self.eos_id = self.tokenizer.convert_tokens_to_ids("[EOS]")
        self.srep_id = self.tokenizer.convert_tokens_to_ids("[S_REP]")
        assert self.eos_id != self.tokenizer.unk_token_id, "Tokenizer must include [EOS]"
        assert self.srep_id != self.tokenizer.unk_token_id, "Tokenizer must include [S_REP]"

        # Embeddings in token space (d_model)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # STM positional encoding in memory space (srep_dim)
        if cfg.use_stm_positional:
            self.stm_pos_encoder = STMPositionalEncoding(cfg.srep_dim, cfg.stm_positional_weight)
        else:
            self.stm_pos_encoder = None

        # Transformer blocks
        self.blocks = nn.ModuleList([SentenceBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # LM head (tied to token embedding)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # ---- S_REP head and projections ----
        # S_REP head (SimCLR-style MLP) produces srep_dim (also memory K/V dim)
        self.sentence_head = _make_srep_head(cfg)

        # Token features -> memory space for retrieval/query
        self.query_proj = nn.Linear(cfg.d_model, cfg.srep_dim)
        self._use_query_proj = (cfg.d_model != cfg.srep_dim)

        # Project S_REP (memory space) back to token space for alignment loss
        self.srep_to_token = nn.Linear(cfg.srep_dim, cfg.d_model)
        self._use_srep_to_token = (cfg.srep_dim != cfg.d_model)

        # Optional small attentive pooling (learned queries in model space).
        # IMPORTANT: This pooling is ADDED to [S_REP] (residual), not a replacement.
        if cfg.use_attentive_pool and cfg.attn_pool_n_queries > 0:
            self.attn_pool_queries = nn.Parameter(
                torch.randn(cfg.attn_pool_n_queries, cfg.d_model)
            )
        else:
            self.attn_pool_queries = None

        # S_REP dropout for diversity during training
        self.srep_dropout = nn.Dropout(cfg.srep_dropout)

        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _causal_mask(L: int, device: torch.device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def _build_stm_tensor(self, stm_lists: List[List[torch.Tensor]], device: torch.device
                         ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build STM tensors with separate KEYS (with positional) and VALUES (raw) in srep_dim.
        Returns: (k_stm_vals, k_stm_keys, k_stm_mask)
        """
        if not stm_lists:
            return None, None, None

        max_k = max((len(arr) for arr in stm_lists), default=0)
        if max_k == 0:
            return None, None, None

        k_stm_val_rows = []
        k_stm_key_rows = []
        k_stm_masks = []

        for arr in stm_lists:
            if arr:
                t = torch.stack(arr, dim=0)  # [K_i, srep_dim]

                if self.stm_pos_encoder is not None:
                    positions = torch.arange(t.size(0), device=device)
                    pos_enc = self.stm_pos_encoder(positions)  # [K_i, srep_dim]
                    t_keys = t + pos_enc
                else:
                    t_keys = t

                K_i = t.size(0)
                if K_i < max_k:
                    t = F.pad(t, (0, 0, 0, max_k - K_i))
                    t_keys = F.pad(t_keys, (0, 0, 0, max_k - K_i))

                mask = torch.ones(max_k, dtype=torch.bool, device=device)
                mask[:K_i] = False
            else:
                t = torch.zeros(max_k, self.cfg.srep_dim, device=device)
                t_keys = t.clone()
                mask = torch.ones(max_k, dtype=torch.bool, device=device)

            k_stm_val_rows.append(t)
            k_stm_key_rows.append(t_keys)
            k_stm_masks.append(mask)

        k_stm_vals = torch.stack(k_stm_val_rows, dim=0)  # [B, K_max, srep_dim]
        k_stm_keys = torch.stack(k_stm_key_rows, dim=0)  # [B, K_max, srep_dim]
        k_stm_mask = torch.stack(k_stm_masks, dim=0)     # [B, K_max]
        return k_stm_vals, k_stm_keys, k_stm_mask

    def iter_document_steps_fixed(
        self,
        documents: List[Dict[str, Any]],
        ltm: Optional["LongTermMemoryGPU"] = None,
        warmup_weight: float = 1.0,
        collect_debug: bool = False,
        debug_max_samples: int = 1,
        # cheap STM grad probes
        collect_stm_grad_refs: bool = False,
        stm_grad_limit: int = 100,
        # component grad probes (cross/self/ffn/layernorm) at [S_REP]
        collect_comp_grad_refs: bool = False,
        comp_grad_limit: int = 100,
        # cadence (based on caller's global step index)
        global_step_start: Optional[int] = None,
        stm_grad_every: int = 0,
        comp_grad_every: int = 0,
        # limit number of rows for component grad sampling
        comp_grad_doc_limit: int = 2,
        # debug emission for LTM detail
        collect_ltm_debug: bool = False,
        ltm_debug_topk: int = 5,
        # context-dropout override for this call (schedule sets this per step in the outer loop)
        context_dropout_now: Optional[float] = None,
        need_alignment_features: bool = False,
        need_stm_means: bool = False,
    ):
        """
        Same API as before + 'context_dropout_now'.
        Retrieval queries are explicitly computed from dropout-free features.
        STM entries are stored without detach so grads flow from future token prediction.
        """
        device = next(self.parameters()).device
        L = self.cfg.max_position_embeddings

        pos_ids_full = torch.arange(L, device=device).unsqueeze(0)
        attn_mask_tri = self._causal_mask(L, device)

        num_docs = len(documents)
        stms: List[Deque[Tuple[torch.Tensor, int]]] = [deque() for _ in range(num_docs)]
        stm_texts: List[Deque[str]] = [deque() for _ in range(num_docs)]

        dbg_docs = set(range(min(num_docs, debug_max_samples))) if collect_debug else set()
        doc_lengths = [len(doc['sentences']) for doc in documents]
        T_max = max(doc_lengths) if doc_lengths else 0

        for doc in documents:
            sent_list = doc.get('sentences', [])
            attn_list = doc.get('attention_masks', [])
            if not sent_list:
                doc.setdefault('sentences_tensor', torch.empty(0, L, dtype=torch.long))
                doc.setdefault('attention_tensor', torch.empty(0, L, dtype=torch.long))
                continue
            cached = doc.get('sentences_tensor', None)
            if cached is None or cached.size(0) != len(sent_list):
                doc['sentences_tensor'] = torch.stack(sent_list, dim=0)
                doc['attention_tensor'] = torch.stack(attn_list, dim=0)

        emitted_steps = 0
        base_ctx_drop = float(self.cfg.context_dropout if context_dropout_now is None else context_dropout_now)
        need_visible_texts = collect_debug or collect_ltm_debug
        need_alignment_features = bool(need_alignment_features)
        need_stm_means = bool(need_stm_means or collect_debug)

        def _cat_mem(stm_k, stm_v, stm_m, ltm_k, ltm_v, ltm_m):
            if stm_k is None or stm_v is None:
                return ltm_k, ltm_v, ltm_m
            if ltm_k is None or ltm_v is None:
                return stm_k, stm_v, stm_m
            k = torch.cat([stm_k, ltm_k], dim=1)
            v = torch.cat([stm_v, ltm_v], dim=1)
            m = torch.cat([stm_m, ltm_m], dim=1)
            return k, v, m

        # ---- Build retrieval query from DROP-OUT FREE token features ----
        def _build_query_from_dropout_free_features(
            ids_: torch.Tensor,
            am_: torch.Tensor,
            layer_ln: nn.LayerNorm
        ) -> torch.Tensor:
            """
            Compute token features WITHOUT any dropout or context-mask effects:
              q_src = ln( tok_emb(ids) + pos_emb )  -> mean over visible tokens -> project to srep_dim -> L2 normalize
            """
            with torch.no_grad():
                B = ids_.size(0)
                pos_ids = pos_ids_full.expand(B, L)
                # no module dropout and NO context-dropout: raw embeddings only
                h0 = self.tok_emb(ids_) + self.pos_emb(pos_ids)
                q_src = layer_ln(h0)

                visible = (am_ == 1) & (ids_ != self.srep_id) & (ids_ != self.eos_id)
                denom = visible.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
                q_mean = (q_src * visible.unsqueeze(-1).float()).sum(dim=1) / denom  # [B, d_model]
                if self._use_query_proj:
                    q = self.query_proj(q_mean)                                       # [B, srep_dim]
                else:
                    q = q_mean
                q = F.normalize(q, p=2, dim=1)
            return q

        for t in range(T_max):
            rows_idx = [d for d in range(num_docs) if t < len(documents[d]['sentences'])]
            if not rows_idx:
                continue

            ids_cpu = torch.stack([documents[d]['sentences_tensor'][t] for d in rows_idx], dim=0)
            am_cpu = torch.stack([documents[d]['attention_tensor'][t] for d in rows_idx], dim=0)
            ids = ids_cpu.to(device, non_blocking=True)
            am  = am_cpu.to(device, non_blocking=True)
            step_B = ids.size(0)

            current_global_step = (global_step_start if global_step_start is not None else 0) + emitted_steps
            collect_stm_this_step  = bool(collect_stm_grad_refs  and (stm_grad_every  <= 0 or (current_global_step % stm_grad_every  == 0)))
            collect_comp_this_step = bool(collect_comp_grad_refs) and (comp_grad_every > 0 and (current_global_step % comp_grad_every == 0))
            force_first_sample = (global_step_start == 0 and emitted_steps == 0)
            collect_comp_this_step = collect_comp_this_step or force_first_sample

            # Embeddings in token space (training path uses embedding dropout)
            h = self.drop(self.tok_emb(ids) + self.pos_emb(pos_ids_full.expand(step_B, L)))

            # Context dropout (scheduled) inside current sentence (training only); EXCLUDED from retrieval query
            if self.training and base_ctx_drop > 0.0:
                keep_mask = (am == 1) & (ids != self.srep_id) & (ids != self.eos_id)
                if keep_mask.any():
                    rand = torch.rand_like(keep_mask, dtype=torch.float32)
                    drop = (rand < base_ctx_drop) & keep_mask
                    if drop.any():
                        h = h.masked_fill(drop.unsqueeze(-1), 0.0)

            key_pad_mask = (am == 0)
            attn_mask = attn_mask_tri

            sent_texts_step = [documents[d]['sentence_texts'][t] for d in rows_idx]
            def _doc_key(ex): return ex.get("original_doc_id", ex.get("doc_id"))
            sent_doc_ids_step = [_doc_key(documents[d]) for d in rows_idx]

            # Visible text prefix (for debug)
            visible_texts_step: List[str] = []
            if need_visible_texts:
                with torch.no_grad():
                    srep_positions_mask = (ids == self.srep_id)
                    token_mask_for_query = (am == 1) & (~srep_positions_mask)
                    for i in range(step_B):
                        toks = ids[i][token_mask_for_query[i]].tolist()
                        visible_texts_step.append(self.tokenizer.decode(toks, clean_up_tokenization_spaces=False))

            ltm_stats_by_block_step: Dict[int, Dict] = {}
            ltm_debug_by_block_step: Dict[int, List[Dict[str, Any]]] = {}
            debug_records_step: List[Dict[str, Any]] = []
            query_comparison_stats: Dict[int, Dict[str, float]] = {}
            actual_stm_sizes = [len(stms[d]) for d in rows_idx]

            use_memory = not self.cfg.debug_no_memory

            stm_grad_refs_step: List[torch.Tensor] = []

            if use_memory:
                stm_lists: List[List[torch.Tensor]] = []
                for d in rows_idx:
                    arr = [tensor for tensor, _ in stms[d]]
                    stm_lists.append(arr)
                any_stm = any(len(arr) > 0 for arr in stm_lists)
                if any_stm:
                    stm_vals_cached, stm_keys_cached, stm_mask_cached = self._build_stm_tensor(stm_lists, device)
                else:
                    stm_vals_cached = stm_keys_cached = stm_mask_cached = None
                if collect_stm_this_step and stm_grad_limit > 0:
                    budget = int(stm_grad_limit)
                    for arr in stm_lists:
                        if budget <= 0:
                            break
                        take = min(len(arr), budget)
                        for ref in arr[:take]:
                            if ref is not None and (ref.requires_grad or getattr(ref, "grad_fn", None) is not None):
                                ref.retain_grad()
                                stm_grad_refs_step.append(ref)
                        budget -= take
            else:
                stm_lists = [[] for _ in rows_idx]
                stm_vals_cached = stm_keys_cached = stm_mask_cached = None

            if collect_debug:
                for i, d in enumerate(rows_idx):
                    if d in dbg_docs:
                        debug_records_step.append({
                            "doc_id": documents[d]['doc_id'],
                            "doc_idx": d,
                            "sentence_index": t,
                            "sentence_text": sent_texts_step[i],
                            "stm_texts": list(stm_texts[d]),
                            "stm_size": len(stms[d]),
                            "ltm_by_block": []
                        })

            # STM means BEFORE adding current S_REP (in memory space), plus token projection for alignment
            stm_means_mem: Optional[torch.Tensor] = None
            stm_means_token: Optional[torch.Tensor] = None
            if need_stm_means and stm_lists:
                stm_means_mem_list: List[torch.Tensor] = []
                with torch.no_grad():
                    for arr in stm_lists:
                        if arr:
                            m = torch.stack(arr, dim=0).mean(dim=0)
                            m = F.normalize(m, p=2, dim=0)
                        else:
                            m = torch.zeros(self.cfg.srep_dim, device=device)
                        stm_means_mem_list.append(m.detach())
                if stm_means_mem_list:
                    stm_means_mem = torch.stack(stm_means_mem_list, dim=0)
                    if self._use_srep_to_token:
                        stm_means_token = self.srep_to_token(stm_means_mem)
                    else:
                        stm_means_token = stm_means_mem

            # First [S_REP] positions per row
            srep_pos_list: List[int] = []
            for i in range(step_B):
                pos = (ids[i] == self.srep_id).nonzero(as_tuple=False).squeeze(-1)
                srep_pos_list.append(int(pos[0].item()) if pos.numel() > 0 else 0)
            srep_positions = torch.as_tensor(srep_pos_list, device=device, dtype=torch.long)

            comp_grad_refs_step: Optional[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]] = {
                'cross_attn': [], 'self_attn': [], 'ffn': [], 'layernorm': []
            } if collect_comp_this_step else None

            # ---- per-layer processing ----
            for layer_idx, block in enumerate(self.blocks):
                use_memory = not self.cfg.debug_no_memory

                k_memory_keys = k_memory_vals = k_memory_mask = None
                k_stm_keys = k_stm_vals = k_stm_mask = None
                k_ltm_keys = k_ltm_vals = k_ltm_mask = None

                if use_memory:
                    k_stm_vals = stm_vals_cached
                    k_stm_keys = stm_keys_cached
                    k_stm_mask = stm_mask_cached

                    # --- Optional LTM retrieval (queries built WITHOUT dropout)
                    if (
                        (not self.cfg.debug_stm_only)
                        and self.cfg.use_long_term_memory
                        and (layer_idx >= self.cfg.no_ltm_for_first_k_blocks)
                        and (ltm is not None)
                        and (warmup_weight > 0.0)
                    ):
                        # Build dropout-free queries from raw embeddings (per requirement #3)
                        q_feat = _build_query_from_dropout_free_features(ids, am, block.ln_self)  # [B, srep_dim]

                        # Hybrid: mix current STM mean when available (in memory space)
                        q_hybrid = q_feat.clone()
                        if self.cfg.ltm_query_mode in ("hybrid", "both"):
                            for i, arr in enumerate(stm_lists):
                                if arr:
                                    stm_mean = torch.stack(arr, dim=0).mean(dim=0)
                                    stm_mean = F.normalize(stm_mean, p=2, dim=0)
                                    q_hybrid[i] = F.normalize(0.5 * (q_hybrid[i] + stm_mean), p=2, dim=0)

                        is_doc_scope = isinstance(ltm, DocumentLongTermMemory)

                        def _retrieve(query_tensor):
                            if is_doc_scope:
                                return ltm.retrieve(
                                    query_tensor, sent_doc_ids_step,
                                    top_k=self.cfg.ltm_top_k,
                                    min_sim=self.cfg.ltm_min_sim,
                                    min_matches=self.cfg.ltm_min_matches
                                )
                            else:
                                return ltm.retrieve_gpu(
                                    query_tensor,
                                    top_k=self.cfg.ltm_top_k,
                                    min_sim=self.cfg.ltm_min_sim,
                                    min_matches=self.cfg.ltm_min_matches
                                )

                        # In "both", gather stats on tokens-only vs hybrid, but use HYBRID for actual retrieval
                        if self.cfg.ltm_query_mode == "both":
                            _, _, _, stats_tokens = _retrieve(q_feat)
                            k_ltm_vals, k_ltm_mask, ltm_texts, stats = _retrieve(q_hybrid)
                            _, _, _, stats_hybrid = _retrieve(q_hybrid)
                            query_comparison_stats[layer_idx] = {
                                'tokens_first_sim': stats_tokens.get('first_sim', 0.0),
                                'hybrid_first_sim': stats_hybrid.get('first_sim', 0.0),
                                'tokens_mean_topk': stats_tokens.get('mean_topk', 0.0),
                                'hybrid_mean_topk': stats_hybrid.get('mean_topk', 0.0),
                            }
                        elif self.cfg.ltm_query_mode == "hybrid":
                            k_ltm_vals, k_ltm_mask, ltm_texts, stats = _retrieve(q_hybrid)
                        else:  # "tokens_only"
                            k_ltm_vals, k_ltm_mask, ltm_texts, stats = _retrieve(q_feat)

                        # keys = values for memory bank
                        k_ltm_keys = k_ltm_vals

                        # Doc/global-scope post-processing: dedupe with STM and doc-early-skip
                        items_by_row_from_stats = stats.get("items_by_row", [[] for _ in range(step_B)])
                        stats_used_row_flags = stats.get("used_row_flags", [False for _ in range(step_B)])

                        if (k_ltm_vals is not None) and (k_ltm_vals.size(1) > 0):
                            if is_doc_scope:
                                for i, d in enumerate(rows_idx):
                                    if actual_stm_sizes[i] < self.cfg.max_sentences_in_short_term:
                                        k_ltm_mask[i, :] = True
                                        ltm_texts[i] = []
                            for i, d in enumerate(rows_idx):
                                if k_ltm_mask.size(1) == 0 or not ltm_texts[i]:
                                    continue
                                if len(stm_texts[d]) == 0:
                                    continue
                                stm_set = set(stm_texts[d])
                                for j, txt in enumerate(ltm_texts[i]):
                                    if txt in stm_set:
                                        k_ltm_mask[i, j] = True

                            q_used = q_hybrid if self.cfg.ltm_query_mode in ("hybrid", "both") else q_feat
                            per_row_first, per_row_mean, per_row_used = [], [], []
                            with torch.no_grad():
                                for i in range(step_B):
                                    used_mask = ~k_ltm_mask[i]
                                    n_used = int(used_mask.sum().item())
                                    if n_used > 0:
                                        sims_i = k_ltm_vals[i, used_mask] @ q_used[i]
                                        per_row_first.append(float(sims_i.max().item()))
                                        per_row_mean.append(float(sims_i.mean().item()))
                                        per_row_used.append(1.0 if n_used >= self.cfg.ltm_min_matches else 0.0)
                                    else:
                                        per_row_first.append(0.0); per_row_mean.append(0.0); per_row_used.append(0.0)
                                stats['first_sim']  = float(torch.tensor(per_row_first).mean().item())
                                stats['mean_topk']  = float(torch.tensor(per_row_mean).mean().item())
                                stats['mean_valid'] = stats['mean_topk']
                                stats['used_ratio'] = float(torch.tensor(per_row_used).mean().item())

                        ltm_stats_by_block_step[layer_idx] = stats

                        if collect_debug and debug_records_step:
                            for i in range(min(len(debug_records_step), len(ltm_texts))):
                                if ltm_texts[i]:
                                    debug_records_step[i]["ltm_by_block"].append({
                                        "block": layer_idx,
                                        "texts": ltm_texts[i][:5]
                                    })

                        if collect_ltm_debug:
                            per_row_debug: List[Dict[str, Any]] = []
                            for i, d in enumerate(rows_idx):
                                row_items = items_by_row_from_stats[i] if i < len(items_by_row_from_stats) else []
                                items_dbg = []
                                early_skip = (is_doc_scope and (actual_stm_sizes[i] < self.cfg.max_sentences_in_short_term))
                                stm_set = set(stm_texts[d]) if len(stm_texts[d]) > 0 else set()
                                selected_count = 0
                                for it in row_items:
                                    status = "selected"
                                    if not it.get('initial_valid', False):
                                        status = "below_min_sim"
                                    elif early_skip:
                                        status = "stm_not_full"
                                    elif it['text'] in stm_set:
                                        status = "duplicate_in_stm"
                                    selected = (status == "selected")
                                    if selected:
                                        selected_count += 1
                                    items_dbg.append({
                                        'rank': int(it.get('rank', 0)),
                                        'sim': float(it.get('sim', 0.0)),
                                        'text': it.get('text', ""),
                                        'initial_valid': bool(it.get('initial_valid', False)),
                                        'final_status': status,
                                        'selected': bool(selected),
                                    })
                                used_post = (selected_count >= self.cfg.ltm_min_matches) and (not early_skip)
                                if not used_post:
                                    for it in items_dbg:
                                        if it['final_status'] == "selected":
                                            it['final_status'] = "not_selected_insufficient_matches"
                                            it['selected'] = False
                                items_dbg_sorted = sorted(items_dbg, key=lambda x: x['rank'])[:max(1, int(ltm_debug_topk))]
                                per_row_debug.append({
                                    'ltm_used_pre': bool(stats_used_row_flags[i]) if i < len(stats_used_row_flags) else None,
                                    'ltm_used_post': bool(used_post),
                                    'ltm_early_skip': bool(early_skip),
                                    'items': items_dbg_sorted
                                })
                            ltm_debug_by_block_step[layer_idx] = per_row_debug

                    # ---- Concatenate STM + (scaled) LTM ----
                    if k_stm_keys is not None and k_stm_vals is not None:
                        if k_ltm_vals is not None and k_ltm_vals.size(1) > 0:
                            if warmup_weight < 1.0:
                                k_ltm_keys = k_ltm_keys * warmup_weight
                                k_ltm_vals = k_ltm_vals * warmup_weight
                            k_memory_keys, k_memory_vals, k_memory_mask = _cat_mem(
                                k_stm_keys, k_stm_vals, k_stm_mask,
                                k_ltm_keys, k_ltm_vals, k_ltm_mask
                            )
                        else:
                            k_memory_keys, k_memory_vals, k_memory_mask = k_stm_keys, k_stm_vals, k_stm_mask
                    elif k_ltm_vals is not None and k_ltm_vals.size(1) > 0:
                        if warmup_weight < 1.0:
                            k_ltm_keys = k_ltm_keys * warmup_weight
                            k_ltm_vals = k_ltm_vals * warmup_weight
                        k_memory_keys, k_memory_vals, k_memory_mask = k_ltm_keys, k_ltm_vals, k_ltm_mask

                # --- Forward through this block
                if collect_comp_this_step:
                    h, comp_full = block(
                        h, attn_mask, key_pad_mask,
                        k_memory_keys, k_memory_vals, k_memory_mask,
                        use_memory=use_memory, memory_weight=warmup_weight,
                        srep_positions=srep_positions,
                        return_srep_deltas=True
                    )
                    if comp_grad_refs_step is not None:
                        n_rows = step_B if comp_grad_doc_limit <= 0 else min(step_B, comp_grad_doc_limit)
                        if n_rows > 0:
                            sr_idx = srep_positions[:n_rows].detach()
                            for key in ('cross_attn', 'self_attn', 'ffn', 'layernorm'):
                                parent = comp_full.get(key, None)
                                if parent is None:
                                    continue
                                if isinstance(parent, torch.Tensor) and parent.requires_grad:
                                    parent.retain_grad()
                                comp_grad_refs_step[key].append((parent, sr_idx))
                else:
                    h = block(
                        h, attn_mask, key_pad_mask,
                        k_memory_keys, k_memory_vals, k_memory_mask,
                        use_memory=use_memory, memory_weight=warmup_weight
                    )

            # ===== end per-layer loop =====

            # Final projection and logits (token space)
            h = self.ln_f(h)
            step_logits = self.lm_head(h)
            step_token_hiddens = h

            # S_REPs, adjacency pairs, norm penalties
            srep_embs_step: List[torch.Tensor] = []
            srep_embs_token_step: Optional[List[torch.Tensor]] = [] if need_alignment_features else None
            srep_raw_norms: List[float] = []
            srep_norm_penalties: List[torch.Tensor] = []
            adjacency_sreps_list: List[torch.Tensor] = []
            adjacency_doc_ids_list: List[str] = []

            sqrt_d = math.sqrt(float(self.cfg.d_model))

            for i, d in enumerate(rows_idx):
                srep_pos = (ids[i] == self.srep_id).nonzero(as_tuple=False).squeeze(-1)
                if srep_pos.numel() == 0:
                    continue
                srep_idx = int(srep_pos[0].item())

                # Base [S_REP] hidden (token space)
                srep_hidden = h[i, srep_idx]
                if self.training:
                    srep_hidden = self.srep_dropout(srep_hidden)

                # ---- ADD attentive pooling residual (learned queries in model space) ----
                if self.cfg.use_attentive_pool and self.attn_pool_queries is not None:
                    valid = (am[i] == 1) & (ids[i] != self.srep_id) & (ids[i] != self.eos_id)
                    if valid.any():
                        tok_feats = h[i, valid]  # [T, d_model]
                        # scores: [T, nQ] = tok_feats @ queries^T
                        scores = tok_feats @ self.attn_pool_queries.t()
                        scores = scores / sqrt_d
                        # mask invalid tokens (though tok_feats already filtered)
                        # normalize across T for each query
                        attn = torch.softmax(scores, dim=0)  # [T, nQ]
                        # pooled per query: [d_model, nQ] = tok_feats^T @ attn
                        pooled_mat = tok_feats.t() @ attn
                        # mean pool across queries -> [d_model]
                        pooled = pooled_mat.mean(dim=1)
                        # residual add into [S_REP]
                        srep_hidden = srep_hidden + pooled

                # Head into memory space (srep_dim)
                svec_raw = self.sentence_head(srep_hidden)

                raw_norm = torch.norm(svec_raw, p=2)
                srep_raw_norms.append(raw_norm.item())

                lower = self.cfg.srep_norm_target - self.cfg.srep_norm_margin
                upper = self.cfg.srep_norm_target + self.cfg.srep_norm_margin
                if raw_norm < lower:
                    penalty = (lower - raw_norm) ** 2
                elif raw_norm > upper:
                    penalty = (raw_norm - upper) ** 2
                else:
                    penalty = torch.tensor(0.0, device=device)
                srep_norm_penalties.append(penalty)

                svec = F.normalize(svec_raw, p=2, dim=0) * self.cfg.srep_norm_target   # memory (srep_dim)
                if need_alignment_features:
                    if self._use_srep_to_token:
                        svec_token = self.srep_to_token(svec)  # token space (d_model)
                    else:
                        svec_token = svec

                # adjacency source
                prev_tensor = stms[d][0][0] if len(stms[d]) > 0 else None
                if prev_tensor is not None:
                    adjacency_sreps_list.append(prev_tensor)
                    adjacency_sreps_list.append(svec)
                    unique_row_id = f"{sent_doc_ids_step[i]}::row{i}"
                    adjacency_doc_ids_list.extend([unique_row_id, unique_row_id])

                # ---- IMPORTANT: keep STM entries with grads (NO detach) ----
                stms[d].appendleft((svec, t))
                stm_texts[d].appendleft(sent_texts_step[i])

                if len(stms[d]) > self.cfg.max_sentences_in_short_term:
                    old_tensor, _old_idx = stms[d].pop()
                    _ = stm_texts[d].pop()
                    del old_tensor

                srep_embs_step.append(svec)
                if need_alignment_features and srep_embs_token_step is not None:
                    srep_embs_token_step.append(svec_token)

            srep_embs_tensor = (
                torch.stack(srep_embs_step, dim=0) if srep_embs_step
                else torch.zeros(0, self.cfg.srep_dim, device=device)
            )
            srep_embs_token_tensor: Optional[torch.Tensor] = None
            if need_alignment_features and srep_embs_token_step:
                srep_embs_token_tensor = torch.stack(srep_embs_token_step, dim=0)
            adjacency_sreps_tensor = (
                torch.stack(adjacency_sreps_list, dim=0) if adjacency_sreps_list
                else torch.zeros(0, self.cfg.srep_dim, device=device)
            )

            if srep_norm_penalties:
                srep_norm_reg_loss = torch.stack(srep_norm_penalties).mean() * self.cfg.srep_norm_reg_weight
            else:
                srep_norm_reg_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # If we didn’t get enough STM refs earlier, fall back to the freshly appended S_REPs
            if collect_stm_this_step and stm_grad_limit > 0 and stm_grad_refs_step:
                pass  # already collected from existing STM
            elif collect_stm_this_step and stm_grad_limit > 0 and not stm_grad_refs_step:
                budget = int(stm_grad_limit)
                for d in rows_idx:
                    if budget <= 0:
                        break
                    if len(stms[d]) > 0:
                        ref = stms[d][0][0]
                        if ref is not None and (ref.requires_grad or getattr(ref, "grad_fn", None) is not None):
                            ref.retain_grad()
                            stm_grad_refs_step.append(ref)
                            budget -= 1

            out_rec = {
                "logits": step_logits,
                "input_ids": ids,
                "key_pad_mask": key_pad_mask,
                "srep_embs": srep_embs_tensor,                # memory space (srep_dim)
                "srep_texts": sent_texts_step,
                "srep_doc_ids": sent_doc_ids_step,
                "srep_norm_reg_loss": srep_norm_reg_loss,
                "srep_raw_norms": srep_raw_norms,
                "ltm_stats_by_block": ltm_stats_by_block_step,
                "debug_records_step": debug_records_step if collect_debug else [],
                "actual_stm_sizes": actual_stm_sizes,
                "query_comparison": query_comparison_stats,
                "adjacency_sreps": adjacency_sreps_tensor,    # srep_dim
                "adjacency_doc_ids": adjacency_doc_ids_list,
                "token_hiddens": step_token_hiddens,          # d_model
                "num_adjacent_pairs": len(adjacency_doc_ids_list) // 2,
                "visible_texts": visible_texts_step,
                "ltm_debug_by_block": ltm_debug_by_block_step if collect_ltm_debug else {},
            }

            if srep_embs_token_tensor is not None:
                out_rec["srep_embs_token"] = srep_embs_token_tensor
            if stm_means_token is not None:
                out_rec["stm_means"] = stm_means_token
            if stm_means_mem is not None:
                out_rec["stm_means_mem"] = stm_means_mem

            if collect_stm_this_step and stm_grad_limit > 0 and stm_grad_refs_step:
                out_rec["stm_grad_refs"] = stm_grad_refs_step
                out_rec["stm_grad_step"] = int(current_global_step)

            if collect_comp_this_step and comp_grad_refs_step is not None:
                if any(len(v) > 0 for v in comp_grad_refs_step.values()):
                    out_rec["comp_grad_refs"] = comp_grad_refs_step
                    out_rec["comp_grad_step"] = int(current_global_step)

            yield out_rec
            emitted_steps += 1


# keep alias
def iter_document_steps(self, *args, **kwargs):
    return self.iter_document_steps_fixed(*args, **kwargs)
