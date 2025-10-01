"""
Dataset Creation Pipeline (Slimmed & Correct)
- Only '= Title =' starts a new document (single '=' per edge after trimming spaces).
- Parquet (.parquet/.pq), JSONL, and generic text files (.train/.txt/.md/...) supported.
- No regex boundary detectors beyond top-level titles.
- Includes: extraction, 3-bucket split, sentence processing, chunking, and analysis helpers.
"""

import os
import re
import json
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
from transformers import PreTrainedTokenizer

# Parquet support (same import name your code used before)
import pandas as _pd

logger = logging.getLogger("create_datasets")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------

def _basename_only(x: str) -> str:
    try:
        return os.path.basename(x)
    except Exception:
        return str(x)

def _slugify(text: str, max_len: int = 80) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len] if s else "doc"

def _is_top_title(line: str) -> bool:
    """
    True iff the line is a top-level article title like '= Title ='
    (exactly one '=' at each edge when spaces are ignored).
    Rejects '== ... ==' and spaced variants '= = ... = ='.
    """
    if not line:
        return False
    t = str(line).strip()
    if not (t.startswith("=") and t.endswith("=")):
        return False
    # collapse spaces to detect spaced '==' variants
    t_nospace = t.replace(" ", "")
    # reject multi '=' at either edge: '==Title==' or '===Title==='
    if t_nospace.startswith("==") or t_nospace.endswith("=="):
        return False
    # robust: exactly two '=' total (one at start, one at end)
    if t_nospace.count("=") != 2:
        return False
    # ensure non-empty inner title
    inner = t.strip("=").strip()
    return len(inner) > 0

def _title_text(line: str) -> str:
    """Return the inner title text of '= Title =' (without surrounding '=')."""
    t = (line or "").strip()
    if _is_top_title(t):
        return t[1:-1].strip()
    return (line or "").strip()

def _count_valid_tokens(mask_row) -> int:
    """
    Counts valid tokens in one sentence-row the same way the loss does
    (exclude EOS and [S_REP]). Works for lists or tensors.
    """
    try:
        total = int(mask_row.sum().item() if hasattr(mask_row.sum(), "item") else mask_row.sum())
    except Exception:
        total = sum(int(v) for v in mask_row)
    return max(0, total - 2)

# --------------------------------------------------------------------------------------------
# Document extractors
# --------------------------------------------------------------------------------------------

def _flush_accumulated_doc(
    out: List[Dict[str, Any]],
    current_lines: List[str],
    current_title: Optional[str],
    per_title_counts: Dict[str, int],
    source_path: str,
    prefix: str,
    base: str
):
    """Internal helper to push an accumulated document."""
    if not current_lines:
        return
    text = "\n".join(current_lines).rstrip()
    if not text:
        return

    stub = _slugify(current_title) if current_title else "doc"
    n = per_title_counts.get(stub, 0)
    per_title_counts[stub] = n + 1
    suffix = "" if n == 0 else f"_{n}"
    doc_id = f"{prefix}_{os.path.splitext(base)[0]}_{stub}{suffix}"

    out.append({
        "text": text,
        "source_file": source_path,
        "doc_id": doc_id,
        "char_count": len(text),
        "word_count": len(text.split()),
        "title": current_title or ""
    })

def extract_documents_from_wikitext_parquet_by_title_rows(
    path: str,
    *,
    min_doc_chars: int = 0
) -> List[Dict[str, Any]]:
    """
    Build documents by scanning parquet rows (in order) and splitting ONLY at '= Title =' rows.
    - Title row is kept as the first line of its document.
    - '== Section ==' / '=== ... ===' remain inside the same document.
    - Content before the first top-level title is ignored.
    """
    if _pd is None:
        raise RuntimeError(
            "Reading .parquet requires pandas+pyarrow. Please `pip install pandas pyarrow`."
        )

    try:
        df = _pd.read_parquet(path, engine="pyarrow")
    except Exception as e_pyarrow:
        logger.warning("PyArrow failed for %s (%s). Trying fastparquet…", path, e_pyarrow)
        df = _pd.read_parquet(path, engine="fastparquet")

    if df.shape[0] == 0:
        return []

    # pick text column
    text_col = None
    for c in ("text", "content", "line", "raw", "body"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = str(df.columns[0])

    s = df[text_col]

    def _to_str(x):
        if x is None:
            return ""
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8", "replace")
            except Exception:
                return x.decode("cp1252", "replace")
        return str(x)

    # normalize to strings
    try:
        sample = next((v for v in s.array if v is not None), None) if hasattr(s, "array") \
                 else (s.dropna().iloc[0] if len(s.dropna()) else None)
    except Exception:
        sample = None

    if isinstance(sample, (bytes, bytearray)):
        df[text_col] = s.apply(_to_str)
    else:
        df[text_col] = s.astype("string", copy=False).fillna("")

    # preserve order if index-like col present
    order_col = next((c for c in ("#", "line_no", "idx", "line_index", "row_id") if c in df.columns), None)
    if order_col is not None:
        df = df.sort_values(by=order_col, kind="mergesort")

    base = os.path.basename(path)
    out: List[Dict[str, Any]] = []

    current_lines: List[str] = []
    current_title: Optional[str] = None
    per_title_counts: Dict[str, int] = {}

    for raw in df[text_col].tolist():
        line = "" if raw is None else str(raw)
        if _is_top_title(line):
            _flush_accumulated_doc(out, current_lines, current_title, per_title_counts, path, "pq", base)
            current_title = _title_text(line)
            current_lines = [line.rstrip()]
        else:
            if current_title is not None:
                current_lines.append(line.rstrip())

    _flush_accumulated_doc(out, current_lines, current_title, per_title_counts, path, "pq", base)

    if min_doc_chars and min_doc_chars > 0:
        out = [d for d in out if d["char_count"] >= min_doc_chars]

    logger.info(f"  Parquet (title-scan) → {len(out)} documents from {base}")
    return out

def extract_documents_from_text_by_title_rows(
    path: str,
    *,
    min_doc_chars: int = 0
) -> List[Dict[str, Any]]:
    """
    Read a plain text file and split ONLY at top titles '= Title ='.
    Title line is kept; subsections '==', '===', spaced variants remain inside.
    Content before the first top title is ignored.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252", errors="replace") as f:
            raw = f.read()

    lines = raw.splitlines()
    out: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_title: Optional[str] = None
    per_title_counts: Dict[str, int] = {}

    base = os.path.basename(path)

    for raw_line in lines:
        line = "" if raw_line is None else str(raw_line)
        if _is_top_title(line):
            _flush_accumulated_doc(out, current_lines, current_title, per_title_counts, path, "txt", base)
            current_title = _title_text(line)
            current_lines = [line.rstrip()]
        else:
            if current_title is not None:
                current_lines.append(line.rstrip())

    _flush_accumulated_doc(out, current_lines, current_title, per_title_counts, path, "txt", base)

    if min_doc_chars and min_doc_chars > 0:
        out = [d for d in out if d["char_count"] >= min_doc_chars]

    logger.info(f"  Text (title-scan) → {len(out)} documents from {base}")
    return out

def extract_documents_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    JSONL: one JSON object → one document.
    'text' key preferred; otherwise 'question' + 'answer'; else concatenate string values.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON line {i+1} in {path}")
                continue
            if "text" in data:
                text = data["text"]
            elif "question" in data and "answer" in data:
                text = f"Question: {data['question']} Answer: {data['answer']}"
            else:
                text = " ".join(v for v in data.values() if isinstance(v, str) and v.strip())
            text = (text or "").strip()
            if not text:
                continue
            out.append({
                "text": text,
                "source_file": path,
                "doc_id": f"jsonl_{os.path.basename(path)}_{i}",
                "char_count": len(text),
                "word_count": len(text.split()),
                "title": ""
            })
    logger.info(f"  JSONL → {len(out)} records from {os.path.basename(path)}")
    return out

def extract_all_documents_from_files(
    file_paths: List[str],
    sentence_splitter,                       # kept for API compatibility (unused here)
    use_document_boundaries: bool = True,    # ignored; only title-based split is used
    max_chars_per_doc: int = -1
) -> List[Dict[str, Any]]:
    """
    Unified dispatcher:
      • .parquet/.pq → scan rows, split at '= Title ='
      • .jsonl      → one record → one doc (no title-scan)
      • everything else (.train/.txt/.md/…) → read file, split at '= Title ='
    """
    logger.info("Extracting all documents from source files...")
    out: List[Dict[str, Any]] = []
    total_raw_chars = 0
    total_raw_words = 0

    for path in file_paths:
        base = os.path.basename(path)
        logger.info(f"Reading and parsing: {path}")
        try:
            if path.lower().endswith((".parquet", ".pq")):
                docs = extract_documents_from_wikitext_parquet_by_title_rows(path)
            elif path.lower().endswith(".jsonl"):
                docs = extract_documents_from_jsonl(path)
            else:
                docs = extract_documents_from_text_by_title_rows(path)

            # Optional warning if a single doc is very large (we do NOT auto-subdivide)
            if max_chars_per_doc > 0:
                for d in docs:
                    if d["char_count"] > max_chars_per_doc:
                        logger.warning(
                            f"Document '{d.get('doc_id','')}' length {d['char_count']} exceeds "
                            f"max_chars_per_doc={max_chars_per_doc}; not subdividing by design."
                        )

            out.extend(docs)
            total_raw_chars += sum(d["char_count"] for d in docs)
            total_raw_words += sum(d["word_count"] for d in docs)

        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")

    logger.info("Raw data statistics:")
    logger.info(f"  Total raw characters: {total_raw_chars:,}")
    logger.info(f"  Total raw words (approx): {total_raw_words:,}")
    logger.info(f"Successfully extracted {len(out)} documents")
    if out:
        totc = sum(d["char_count"] for d in out)
        totw = sum(d.get("word_count", len(d["text"].split())) for d in out)
        logger.info(f"  Total document characters: {totc:,}")
        logger.info(f"  Total document words: {totw:,}")
    return out

# --------------------------------------------------------------------------------------------
# Sampling utilities (for quick human inspection)
# --------------------------------------------------------------------------------------------

def write_split_sentence_samples(
    docs: List[Dict[str, Any]],
    out_dir: str,
    split_name: str,
    k: int = 10,
    splitter=None,
    seed: int = 13,
    max_sentences_per_doc: int = 200,
) -> str:
    """
    Like write_split_text_samples, but renders the sentence-split view:
    one sentence per line using the provided splitter.
    Output file: <out_dir>/<split_name>_sample_sentences.txt
    """
    if splitter is None:
        raise ValueError("write_split_sentence_samples requires a `splitter` instance.")
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(seed)
    picks = docs if len(docs) <= k else rng.sample(docs, k)

    path = os.path.join(out_dir, f"{split_name}_sample_sentences.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i, d in enumerate(picks, 1):
            src = _basename_only(d.get("source_file", ""))
            did = d.get("doc_id", f"doc_{i}")
            first_line = (d.get("text") or "").splitlines()[0] if d.get("text") else ""

            f.write("=" * 88 + "\n")
            f.write(f"[{split_name.upper()} DOC {i:02d}] id={did}  src={src}\n")
            f.write(f"{first_line}\n")
            f.write("=" * 88 + "\n")

            sentences = splitter.split_text(d.get("text", "") or "")
            for s_idx, s in enumerate(sentences[:max_sentences_per_doc], 1):
                f.write(f"{s_idx:03d}: {s.strip()}\n")
            if len(sentences) > max_sentences_per_doc:
                f.write(f"... (truncated to {max_sentences_per_doc} sentences)\n")
            f.write("\n")

    logger.info(f"✓ Wrote {len(picks)} sentence-split samples for '{split_name}' → {path}")
    return path


def write_split_text_samples(
    docs: List[Dict[str, Any]],
    out_dir: str,
    split_name: str,
    k: int = 10,
    seed: int = 13
) -> str:
    """
    Write up to k full documents (raw text) to a readable .txt file per split.
    Returns the file path. Used by train.py when explicit splits exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    picks = docs if len(docs) <= k else rng.sample(docs, k)
    path = os.path.join(out_dir, f"{split_name}_sample_docs.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i, d in enumerate(picks, 1):
            first_line = (d.get("text") or "").splitlines()[0] if d.get("text") else ""
            f.write("=" * 88 + "\n")
            f.write(f"[{split_name.upper()} DOC {i:02d}] id={d.get('doc_id','')}  src={_basename_only(d.get('source_file',''))}\n")
            f.write(f"{first_line}\n")
            f.write("=" * 88 + "\n\n")
            f.write((d.get("text") or "").rstrip() + "\n\n")
    logger.info(f"✓ Wrote {len(picks)} sample docs for '{split_name}' → {path}")
    return path

def write_sample_docs(docs: List[Dict[str, Any]], out_path: str, k: int = 10) -> str:
    """
    Write the first k documents to a UTF-8 text file for quick inspection.
    Used by train.py when falling back to internal split.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    k = max(0, int(k))
    with open(out_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs[:k], 1):
            src = _basename_only(d.get("source_file", ""))
            did = d.get("doc_id", f"doc_{i}")
            f.write(f"{'-'*80}\nDOC {i} | {did} | {src}\n{'-'*80}\n")
            f.write((d.get("text", "") or "").rstrip() + "\n\n")
    return out_path

# --------------------------------------------------------------------------------------------
# Split (3-bucket by sentence counts, pre-chunk)
# --------------------------------------------------------------------------------------------

def split_documents_by_length_3buckets(
    docs: List[Dict[str, Any]],
    tokenizer,
    splitter,
    *,
    min_sentences_per_document: int = 2,
    train_val_split: float = 0.9,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Document-level split balanced by 3 length buckets (LOW/MED/HIGH).
    Buckets are computed from the 33% and 66% quantiles of sentence counts
    measured *before chunking*. The split occurs at the document level.
    """
    rng = random.Random(seed)

    # process once to get sentence counts (pre-chunk)
    processed_all: List[Dict[str, Any]] = process_documents_for_training(
        docs, tokenizer, splitter,
        max_sentence_tokens=64,
        min_sentences_per_document=min_sentences_per_document,
        min_sentence_tokens_filter=getattr(splitter, "min_sentence_tokens", None),
    )

    # map doc_id -> sentence_count
    by_id_count: Dict[str, int] = {}
    for ex in processed_all:
        doc_id = str(ex.get("doc_id", ""))
        n = len(ex.get("sentences", []))
        if n >= min_sentences_per_document:
            by_id_count[doc_id] = n

    if not by_id_count:
        return [], []

    # keep only accepted docs
    id_to_doc = {str(d.get("doc_id", "")): d for d in docs}
    kept_docs = [id_to_doc[_id] for _id in by_id_count.keys() if _id in id_to_doc]

    # quantiles
    counts = np.array(list(by_id_count.values()), dtype=float)
    q33 = float(np.quantile(counts, 1/3))
    q66 = float(np.quantile(counts, 2/3))

    def bucket_of(n: int) -> str:
        if n <= q33: return "LOW"
        if n <= q66: return "MED"
        return "HIGH"

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in kept_docs:
        doc_id = str(d.get("doc_id", ""))
        buckets[bucket_of(by_id_count.get(doc_id, 0))].append(d)

    train_docs: List[Dict[str, Any]] = []
    val_docs:   List[Dict[str, Any]] = []

    for bname, group in buckets.items():
        rng.shuffle(group)
        n = len(group)
        n_val = int(round((1.0 - train_val_split) * n))
        if n_val == 0 and n > 0:
            n_val = 1
        val_docs.extend(group[:n_val])
        train_docs.extend(group[n_val:])

    return train_docs, val_docs

# --------------------------------------------------------------------------------------------
# Sentence processing + chunking
# --------------------------------------------------------------------------------------------

def _assert_specials(tokenizer: PreTrainedTokenizer):
    assert tokenizer.convert_tokens_to_ids("[EOS]") != tokenizer.unk_token_id, "[EOS] missing in tokenizer"
    assert tokenizer.convert_tokens_to_ids("[S_REP]") != tokenizer.unk_token_id, "[S_REP] missing in tokenizer"
    assert tokenizer.pad_token_id is not None, "pad_token must be set"

def process_documents_for_training(
    documents: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    sentence_splitter,
    max_sentence_tokens: int = 64,
    min_sentences_per_document: int = 5,
    max_sentences_per_document: Optional[int] = None,
    min_sentence_tokens_filter: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    For each input doc returns one training example with:
      - 'sentences': List[LongTensor(context_len)]
      - 'attention_masks': List[LongTensor(context_len)]
      - 'doc_id', 'num_sentences', 'sentence_texts', 'source_file'
      - 'filtering' footprint (counts of dropped short sentences/tokens)
    """
    _assert_specials(tokenizer)
    eos_id = tokenizer.convert_tokens_to_ids("[EOS]")
    srep_id = tokenizer.convert_tokens_to_ids("[S_REP]")
    pad_id = tokenizer.pad_token_id

    examples: List[Dict[str, Any]] = []
    stats = {
        "total_docs": len(documents),
        "too_few_sentences": 0,
        "truncated_docs": 0,
        "processed": 0,
        "total_sentences": 0,
        "short_sentences_dropped": 0,
        "short_tokens_dropped": 0,
        "total_sentences_raw": 0,
        "total_tokens_raw": 0,
    }

    for doc in tqdm(documents, desc="Processing documents"):
        sentences_raw = sentence_splitter.split_text(doc["text"])

        # per-doc counters
        doc_short_sents = 0
        doc_short_toks = 0
        doc_total_raw_sents = 0
        doc_total_raw_toks = 0

        sentence_ids_list: List[torch.Tensor] = []
        attention_masks_list: List[torch.Tensor] = []
        sentence_texts: List[str] = []

        for sent in sentences_raw:
            tok_ids_raw = tokenizer.encode(sent, add_special_tokens=False)
            raw_len = len(tok_ids_raw)

            doc_total_raw_sents += 1
            doc_total_raw_toks  += raw_len

            # Hard filter: too short?
            if (min_sentence_tokens_filter is not None) and (raw_len < int(min_sentence_tokens_filter)):
                doc_short_sents += 1
                doc_short_toks  += raw_len
                continue

            if raw_len == 0:
                continue

            # cap for training
            if raw_len > max_sentence_tokens:
                tok_ids = tok_ids_raw[:max_sentence_tokens]
            else:
                tok_ids = tok_ids_raw

            # add [EOS] and [S_REP], then pad to target len
            tok_ids = tok_ids + [eos_id, srep_id]
            target_len = max_sentence_tokens + 2
            attn_len = len(tok_ids)
            if attn_len < target_len:
                pad_len = target_len - attn_len
                ids = tok_ids + [pad_id] * pad_len
                mask = [1] * attn_len + [0] * pad_len
            else:
                ids = tok_ids
                mask = [1] * target_len

            sentence_ids_list.append(torch.tensor(ids, dtype=torch.long))
            attention_masks_list.append(torch.tensor(mask, dtype=torch.long))
            sentence_texts.append(sent)

        # dataset-level aggregates
        stats["short_sentences_dropped"] += doc_short_sents
        stats["short_tokens_dropped"]   += doc_short_toks
        stats["total_sentences_raw"]    += doc_total_raw_sents
        stats["total_tokens_raw"]       += doc_total_raw_toks

        if len(sentence_ids_list) < min_sentences_per_document:
            stats["too_few_sentences"] += 1
            continue

        if max_sentences_per_document and len(sentence_ids_list) > max_sentences_per_document:
            sentence_ids_list = sentence_ids_list[:max_sentences_per_document]
            attention_masks_list = attention_masks_list[:max_sentences_per_document]
            sentence_texts = sentence_texts[:max_sentences_per_document]
            stats["truncated_docs"] += 1

        examples.append({
            "sentences": sentence_ids_list,
            "attention_masks": attention_masks_list,
            "doc_id": doc.get("doc_id", "unknown"),
            "num_sentences": len(sentence_ids_list),
            "sentence_texts": sentence_texts,
            "source_file": doc.get("source_file", "unknown"),
            "original_doc_id": doc.get("doc_id", "unknown"),
            "filtering": {
                "short_sentences_dropped": int(doc_short_sents),
                "short_tokens_dropped":   int(doc_short_toks),
                "total_sentences_raw":    int(doc_total_raw_sents),
                "total_tokens_raw":       int(doc_total_raw_toks),
                "min_sentence_tokens":    int(min_sentence_tokens_filter or 0),
            }
        })
        stats["processed"] += 1
        stats["total_sentences"] += len(sentence_ids_list)

    logger.info("Processing statistics:")
    logger.info(f"  Total documents: {stats['total_docs']}")
    logger.info(f"  Processed: {stats['processed']}")
    logger.info(f"  Too few sentences: {stats['too_few_sentences']}")
    logger.info(f"  Truncated docs: {stats['truncated_docs']}")
    logger.info(f"  Total sentences (kept): {stats['total_sentences']}")
    if stats["total_sentences_raw"] > 0:
        pct_s = stats["short_sentences_dropped"] / stats["total_sentences_raw"]
        pct_t = stats["short_tokens_dropped"]   / max(1, stats["total_tokens_raw"])
        logger.info(
            f"  Short-sentence filter (<{min_sentence_tokens_filter} toks): "
            f"dropped {stats['short_sentences_dropped']}/{stats['total_sentences_raw']} ({pct_s:.2%}); "
            f"tokens dropped {stats['short_tokens_dropped']}/{stats['total_tokens_raw']} ({pct_t:.2%})"
        )
    return examples

def chunk_processed_documents(
    processed_docs: List[Dict[str, Any]],
    chunk_size: int = 45,
    overlap_size: int = 15,
    split_name: str = "train"
) -> List[Dict[str, Any]]:
    """
    Overlapping chunker (sentences). stride = chunk_size - overlap_size.
    - Deterministic coverage; final tail is always included.
    """
    logger.info(f"Chunking {split_name} documents (chunk_size={chunk_size}, overlap={overlap_size})...")
    stride = chunk_size - overlap_size
    if stride <= 0:
        raise ValueError(f"Overlap size ({overlap_size}) must be less than chunk size ({chunk_size})")

    chunked_docs: List[Dict[str, Any]] = []
    total_chunks = 0
    docs_unchanged = 0
    docs_chunked = 0

    total_unique_sentences = 0
    total_chunked_sentences = 0

    for doc in tqdm(processed_docs, desc=f"Chunking {split_name}"):
        S = doc["num_sentences"]
        total_unique_sentences += S

        if S <= chunk_size:
            chunked_docs.append(doc)
            docs_unchanged += 1
            total_chunks += 1
            total_chunked_sentences += S
            continue

        docs_chunked += 1

        starts = list(range(0, max(1, S - chunk_size) + 1, stride))
        if starts[-1] + chunk_size < S:
            starts.append(S - chunk_size)

        first_ix = len(chunked_docs)
        for j, start in enumerate(starts):
            end = min(start + chunk_size, S)
            chunk = {
                "sentences":        doc["sentences"][start:end],
                "attention_masks":  doc["attention_masks"][start:end],
                "sentence_texts":   doc["sentence_texts"][start:end],
                "doc_id":           f"{doc['doc_id']}_chunk_{j}",
                "num_sentences":    end - start,
                "source_file":      doc["source_file"],
                "original_doc_id":  doc.get("original_doc_id", doc["doc_id"]),
                "chunk_info": {
                    "chunk_idx": j,
                    "start_sentence": start,
                    "end_sentence": end,
                    "total_chunks_from_doc": -1  # filled below
                }
            }
            chunked_docs.append(chunk)
            total_chunks += 1
            total_chunked_sentences += (end - start)

        total_from_doc = len(starts)
        for ix in range(first_ix, len(chunked_docs)):
            chunked_docs[ix]["chunk_info"]["total_chunks_from_doc"] = total_from_doc

    example_expansion = total_chunks / max(1, len(processed_docs))
    sentence_dup_factor = total_chunked_sentences / max(1, total_unique_sentences)

    logger.info(f"Chunking statistics for {split_name}:")
    logger.info(f"  Original documents: {len(processed_docs)}")
    logger.info(f"  Documents unchanged (≤{chunk_size} sentences): {docs_unchanged}")
    logger.info(f"  Documents chunked: {docs_chunked}")
    logger.info(f"  Total chunks created: {total_chunks}")
    logger.info(f"  Example-count expansion (chunks/docs): {example_expansion:.2f}x")
    logger.info(
        f"  Sentence duplication factor (due to overlap): {sentence_dup_factor:.2f}x  "
        f"(stride={stride}, long-doc asymptote ≈ {chunk_size/stride:.2f}x)"
    )

    chunk_sentence_counts = [c["num_sentences"] for c in chunked_docs]
    if chunk_sentence_counts:
        logger.info(
            f"  Chunk sizes — Mean: {np.mean(chunk_sentence_counts):.1f}, "
            f"Min: {min(chunk_sentence_counts)}, Max: {max(chunk_sentence_counts)}"
        )
    return chunked_docs

# --------------------------------------------------------------------------------------------
# Analysis helpers (full & essentials)
# --------------------------------------------------------------------------------------------

def compute_essential_dataset_stats(
    split_name: str,
    pre_docs: List[Dict[str, Any]],
    post_examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Essentials:
      totals (pre/post), repetition ratios post-chunk,
      filtering footprint (short sentences/tokens).
    """
    # Totals (kept, pre-chunk)
    total_sentences_pre = 0
    total_tokens_pre = 0
    for d in pre_docs:
        sents = d.get("sentences", [])
        masks = d.get("attention_masks", [])
        total_sentences_pre += len(sents)
        for i in range(len(sents)):
            mr = masks[i] if i < len(masks) else []
            total_tokens_pre += _count_valid_tokens(mr)

    # gather filtering info from per-doc metadata
    short_sent_dropped = 0
    short_tok_dropped = 0
    total_sent_raw = 0
    total_tok_raw = 0
    for d in pre_docs:
        filt = d.get("filtering", {})
        short_sent_dropped += int(filt.get("short_sentences_dropped", 0))
        short_tok_dropped  += int(filt.get("short_tokens_dropped", 0))
        total_sent_raw     += int(filt.get("total_sentences_raw", 0))
        total_tok_raw      += int(filt.get("total_tokens_raw", 0))

    if total_sent_raw == 0:
        total_sent_raw = total_sentences_pre + short_sent_dropped
    if total_tok_raw == 0:
        total_tok_raw = total_tokens_pre + short_tok_dropped

    # Totals (post-chunk)
    total_examples_post = len(post_examples)
    total_sentences_post = 0
    total_tokens_post = 0

    # Repetition bookkeeping (post-chunk)
    occ: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ex in post_examples:
        doc_id = str(ex.get("doc_id", ""))
        masks  = ex.get("attention_masks", [])
        texts  = ex.get("sentence_texts", [])
        total_sentences_post += len(masks)
        for i in range(len(masks)):
            mr = masks[i] if i < len(masks) else []
            tok = _count_valid_tokens(mr)
            total_tokens_post += tok
            key = (doc_id, (texts[i] if i < len(texts) else "").strip().lower())
            occ[key].append(tok)

    repeated_sentences_total = 0
    repeated_tokens_total = 0
    for _key, tok_list in occ.items():
        if len(tok_list) > 1:
            repeated_sentences_total += (len(tok_list) - 1)
            repeated_tokens_total += sum(tok_list[1:])

    sentence_repeat_ratio = (repeated_sentences_total / total_sentences_post) if total_sentences_post > 0 else 0.0
    token_repeat_ratio    = (repeated_tokens_total    / total_tokens_post)    if total_tokens_post    > 0 else 0.0

    short_sent_pct = (short_sent_dropped / total_sent_raw) if total_sent_raw > 0 else 0.0
    short_tok_pct  = (short_tok_dropped  / total_tok_raw)  if total_tok_raw  > 0 else 0.0

    return {
        "dataset": split_name,
        "totals": {
            "total_examples_postchunk": int(total_examples_post),
            "total_sentences_prechunk": int(total_sentences_pre),
            "total_sentences_postchunk": int(total_sentences_post),
            "total_tokens_prechunk": int(total_tokens_pre),
            "total_tokens_postchunk": int(total_tokens_post),
        },
        "repetition": {
            "sentence_repeat_ratio_postchunk": float(sentence_repeat_ratio),
            "token_repeat_ratio_postchunk": float(token_repeat_ratio),
        },
        "filtering": {
            "short_sentences_dropped": int(short_sent_dropped),
            "short_sentences_raw":     int(total_sent_raw),
            "short_sentences_drop_pct": float(short_sent_pct),
            "short_tokens_dropped":    int(short_tok_dropped),
            "short_tokens_raw":        int(total_tok_raw),
            "short_tokens_drop_pct":   float(short_tok_pct),
        },
    }

def compute_full_dataset_analysis(
    dataset_name: str,
    examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Full post-chunk analysis:
      - sentences per example stats
      - tokens per sentence stats (excl. EOS/S_REP)
      - composition by source (examples/sentences/tokens)
    """
    from collections import Counter

    stats: Dict[str, Any] = {"dataset": dataset_name}

    # sentences per example
    sent_counts = np.array([ex["num_sentences"] for ex in examples], dtype=np.int32)
    stats["examples"] = {
        "count": int(len(examples)),
        "sentences_per_example": {
            "mean": float(np.mean(sent_counts)) if len(sent_counts) else 0.0,
            "std":  float(np.std(sent_counts)) if len(sent_counts) else 0.0,
            "min":  int(np.min(sent_counts)) if len(sent_counts) else 0,
            "max":  int(np.max(sent_counts)) if len(sent_counts) else 0,
            "p50":  float(np.percentile(sent_counts, 50)) if len(sent_counts) else 0.0,
            "p90":  float(np.percentile(sent_counts, 90)) if len(sent_counts) else 0.0,
            "p95":  float(np.percentile(sent_counts, 95)) if len(sent_counts) else 0.0,
            "p99":  float(np.percentile(sent_counts, 99)) if len(sent_counts) else 0.0,
        }
    }

    # tokens per sentence (exclude EOS,S_REP)
    tok_lens = []
    by_source_examples = Counter()
    by_source_sentences = Counter()
    by_source_tokens = Counter()

    def _basename(p):
        return os.path.basename(p or "unknown")

    for ex in examples:
        src = _basename(ex.get("source_file", "unknown"))
        by_source_examples[src] += 1
        by_source_sentences[src] += ex["num_sentences"]
        for mask in ex["attention_masks"]:
            L = int(mask.sum().item())  # includes EOS,S_REP
            eff = max(0, L - 2)
            tok_lens.append(eff)
            by_source_tokens[src] += eff

    tok_lens = np.array(tok_lens, dtype=np.int32)
    stats["tokens_per_sentence"] = {
        "count": int(tok_lens.size),
        "mean": float(np.mean(tok_lens)) if tok_lens.size else 0.0,
        "std":  float(np.std(tok_lens)) if tok_lens.size else 0.0,
        "min":  int(np.min(tok_lens)) if tok_lens.size else 0,
        "max":  int(np.max(tok_lens)) if tok_lens.size else 0,
        "p50":  float(np.percentile(tok_lens, 50)) if tok_lens.size else 0.0,
        "p90":  float(np.percentile(tok_lens, 90)) if tok_lens.size else 0.0,
        "p95":  float(np.percentile(tok_lens, 95)) if tok_lens.size else 0.0,
        "p99":  float(np.percentile(tok_lens, 99)) if tok_lens.size else 0.0,
    }

    def _as_percent(counter: Dict[str, int]) -> Dict[str, float]:
        total = max(1, sum(counter.values()))
        return {k: 100.0 * v / total for k, v in sorted(counter.items(), key=lambda kv: -kv[1])}

    stats["composition"] = {
        "by_examples_pct":  _as_percent(by_source_examples),
        "by_sentences_pct": _as_percent(by_source_sentences),
        "by_tokens_pct":    _as_percent(by_source_tokens),
    }
    return stats

def write_analysis_files(out_dir: str, split_label: str, cache_key: str, analysis: dict):
    """
    Writes:
      <out_dir>/<split_label>_<cache_key>_analysis.json
      <out_dir>/<split_label>_<cache_key>_analysis.txt
    Renders both full and essentials variants gracefully.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = f"{split_label}_{cache_key}_analysis"
    json_path = os.path.join(out_dir, f"{base}.json")
    txt_path  = os.path.join(out_dir, f"{base}.txt")

    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Pretty text rendering
    def _fmt_stats(stats: dict) -> str:
        if not isinstance(stats, dict) or not stats:
            return "(none)"
        order = ["count","mean","std","min","max","p50","p90","p95","p99"]
        parts = []
        for k in order:
            if k in stats:
                v = stats[k]
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        for k in sorted(stats.keys()):
            if k not in order:
                v = stats[k]
                parts.append(f"{k}={v}")
        return ", ".join(parts) if parts else "(empty)"

    def _fmt_pct_map(mp: dict, indent: int = 4) -> str:
        if not isinstance(mp, dict) or not mp:
            return " " * indent + "(none)\n"
        items = sorted(mp.items(), key=lambda kv: (-kv[1], kv[0]))
        lines = []
        for k, v in items:
            lines.append(" " * indent + f"{k}: {v:.2f}%")
        return "\n".join(lines) + "\n"

    lines = []
    lines.append(f"Dataset: {analysis.get('dataset', split_label)}")
    lines.append(f"Split label: {split_label}")
    lines.append("")

    if "totals" in analysis:
        t = analysis["totals"] or {}
        lines.append("Totals:")
        for k in [
            "total_examples_postchunk",
            "total_sentences_prechunk",
            "total_sentences_postchunk",
            "total_tokens_prechunk",
            "total_tokens_postchunk",
        ]:
            if k in t:
                lines.append(f"  {k}: {t[k]}")
        lines.append("")

    if "repetition" in analysis:
        r = analysis["repetition"] or {}
        lines.append("Repetition:")
        for k in [
            "repeated_sentences_total",
            "repeated_sentences_unique",
            "repetition_ratio",
            "overlap_duplicates_total",
            "sentence_repeat_ratio_postchunk",
            "token_repeat_ratio_postchunk",
        ]:
            if k in r:
                val = r[k]
                lines.append(f"  {k}: {val:.6f}" if isinstance(val, float) else f"  {k}: {val}")
        lines.append("")

    if "filtering" in analysis:
        fsec = analysis["filtering"] or {}
        lines.append("Filtering:")
        for k in [
            "short_sentences_dropped",
            "short_sentences_raw",
            "short_sentences_drop_pct",
            "short_tokens_dropped",
            "short_tokens_raw",
            "short_tokens_drop_pct",
        ]:
            if k in fsec:
                val = fsec[k]
                lines.append(f"  {k}: {val:.6f}" if isinstance(val, float) else f"  {k}: {val}")
        lines.append("")

    ex = analysis.get("examples")
    if isinstance(ex, dict):
        lines.append("Examples:")
        if "count" in ex:
            lines.append(f"  count: {ex['count']}")
        if "sentences_per_example" in ex:
            lines.append("  sentences_per_example: " + _fmt_stats(ex["sentences_per_example"]))
        lines.append("")

    tps = analysis.get("tokens_per_sentence")
    if isinstance(tps, dict):
        lines.append("Tokens per sentence:")
        lines.append("  " + _fmt_stats(tps))
        lines.append("")

    comp = analysis.get("composition")
    if isinstance(comp, dict):
        lines.append("Composition:")
        if "by_examples_pct" in comp:
            lines.append("  by_examples_pct:")
            lines.append(_fmt_pct_map(comp["by_examples_pct"], indent=4))
        if "by_sentences_pct" in comp:
            lines.append("  by_sentences_pct:")
            lines.append(_fmt_pct_map(comp["by_sentences_pct"], indent=4))
        if "by_tokens_pct" in comp:
            lines.append("  by_tokens_pct:")
            lines.append(_fmt_pct_map(comp["by_tokens_pct"], indent=4))

    with open(txt_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return json_path, txt_path

def create_analysis_report(
    dataset_name: str,
    examples: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    A concise human-readable overview; not used by caching but handy for quick checks.
    """
    lines = [f"--- ANALYSIS for {dataset_name} ---\n"]
    if not examples:
        lines.append("No examples found.")
    else:
        num_sentences = [ex['num_sentences'] for ex in examples]
        lines.append(f"Total documents/chunks: {len(examples)}")
        lines.append(f"Total sentences: {sum(num_sentences)}")
        lines.append(f"\nSentences per document/chunk:")
        lines.append(f"  Mean: {np.mean(num_sentences):.1f}")
        lines.append(f"  Std: {np.std(num_sentences):.1f}")
        lines.append(f"  Min: {min(num_sentences)}")
        lines.append(f"  Max: {max(num_sentences)}")
        lines.append(f"  Median: {np.median(num_sentences):.1f}")

        chunked_count = sum(1 for ex in examples if "chunk_info" in ex)
        if chunked_count > 0:
            lines.append(f"\nChunking statistics:")
            lines.append(f"  Chunked documents: {chunked_count}")
            lines.append(f"  Unchunked documents: {len(examples) - chunked_count}")

            original_docs = set()
            for ex in examples:
                if "original_doc_id" in ex:
                    original_docs.add(ex["original_doc_id"])
            lines.append(f"  Original documents: {len(original_docs)}")
            lines.append(f"  Expansion factor: {len(examples) / len(original_docs):.2f}x")

        all_sent_lengths = []
        for ex in examples[:min(100, len(examples))]:
            for mask in ex['attention_masks']:
                length = int(mask.sum().item()) - 2  # exclude [EOS], [S_REP]
                all_sent_lengths.append(length)
        if all_sent_lengths:
            lines.append(f"\nSentence lengths (tokens, sample of {len(all_sent_lengths)} sentences):")
            lines.append(f"  Mean: {np.mean(all_sent_lengths):.1f}")
            lines.append(f"  Std: {np.std(all_sent_lengths):.1f}")
            lines.append(f"  Min: {min(all_sent_lengths)}")
            lines.append(f"  Max: {max(all_sent_lengths)}")

        lines.append("\n[ Composition by source file ]")
        source_counts = defaultdict(int)
        for ex in examples:
            source_counts[os.path.basename(ex.get('source_file', 'unknown'))] += 1
        for fn, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            pct = 100.0 * count / len(examples)
            lines.append(f"  {fn:<30}: {pct:5.1f}% ({count} docs/chunks)")

    report = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Analysis report saved to: {output_path}")
    return report
