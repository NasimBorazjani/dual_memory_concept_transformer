#!/usr/bin/env python
"""Sentence splitter module with token-based length limits."""
import re
import torch
import logging
from typing import List, Union, Optional, Iterator
from transformers import PreTrainedTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenBasedSentenceSplitter:
    """
    A class for splitting text into sentences with TOKEN-BASED length limits.

    This version consistently uses token count (not character count) for all length
    calculations, making it more suitable for language modeling.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        use_model: bool = True,
        model_name: str = "sat-3l",
        sentence_threshold: float = 0.2,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 32,
        max_sentence_tokens: Optional[int] = None,  # Now explicitly token-based
        min_sentence_tokens: Optional[int] = None,  # Now explicitly token-based
        fallback_separators: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the token-based sentence splitter.

        Args:
            tokenizer: The tokenizer to use for token counting
            use_model: Whether to use the SaT model for splitting
            model_name: Name of the SaT model to use
            sentence_threshold: Threshold for sentence boundary detection
            device: Device to run the model on
            batch_size: Batch size for model inference
            max_sentence_tokens: Maximum number of tokens in a sentence (will split longer ones)
            min_sentence_tokens: Minimum number of tokens in a sentence (will filter shorter ones)
            fallback_separators: List of separators for hierarchical splitting
            **kwargs: Additional parameters for backward compatibility
        """
        self.tokenizer = tokenizer
        self.use_model = use_model
        self.model_name = model_name
        self.sentence_threshold = sentence_threshold
        self.batch_size = batch_size
        self.max_sentence_tokens = max_sentence_tokens
        self.min_sentence_tokens = min_sentence_tokens

        # Hierarchical separators for intelligent splitting
        self.fallback_separators = fallback_separators or [
            "...", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", "\t", " "
        ]

        # Regex patterns for sentence splitting
        self.sentence_end_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize SaT model if requested
        self.model = None
        if use_model:
            try:
                from wtpsplit import SaT
                self.model = SaT(model_name)
                if 'cuda' in str(self.device):
                    self.model.half().to(self.device)
                logger.info(f"Successfully loaded SaT model {model_name}")
            except ImportError:
                logger.warning("wtpsplit not available, falling back to regex-based splitting")
                self.use_model = False
            except Exception as e:
                logger.warning(f"Error loading SaT model: {e}. Falling back to regex-based splitting")
                self.use_model = False

        logger.info(f"Initialized TokenBasedSentenceSplitter:")
        logger.info(f"  max_sentence_tokens: {max_sentence_tokens}")
        logger.info(f"  min_sentence_tokens: {min_sentence_tokens}")
        logger.info(f"  use_model: {use_model}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if not text.strip():
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _resplit_by_separator(self, text: str, max_tokens: int, separator: str) -> List[str]:
        """
        Split text by separator while keeping chunks under max_tokens.
        Uses token counting for length limits.
        """
        if separator not in text:
            return [text]

        parts = text.split(separator)
        if len(parts) <= 1:
            return [text]

        result = []
        current_chunk = ""

        for i, part in enumerate(parts[:-1]):
            # Add separator back (except for space separator to avoid double spaces)
            part_with_sep = part + separator if separator != " " else part + " "

            # Check if adding this part would exceed token limit
            test_chunk = current_chunk + part_with_sep
            if self._count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                # Current chunk is full, start a new one
                if current_chunk.strip():
                    result.append(current_chunk.strip())
                current_chunk = part_with_sep

        # Handle the last part
        last_part = parts[-1]
        test_chunk = current_chunk + last_part
        if self._count_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            # Last part doesn't fit, split it off
            if current_chunk.strip():
                result.append(current_chunk.strip())
            current_chunk = last_part

        if current_chunk.strip():
            result.append(current_chunk.strip())

        return result if result else [text]

    def _hierarchical_split(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text using hierarchical approach with token counting.
        """
        if self._count_tokens(text) <= max_tokens:
            return [text]

        logger.debug(f"Text has {self._count_tokens(text)} tokens, splitting (max: {max_tokens})")

        chunks = [text]

        # Try each separator in order
        for separator in self.fallback_separators:
            new_chunks = []
            made_progress = False

            for chunk in chunks:
                chunk_tokens = self._count_tokens(chunk)
                if chunk_tokens <= max_tokens:
                    new_chunks.append(chunk)
                else:
                    # Try splitting this chunk
                    split_chunks = self._resplit_by_separator(chunk, max_tokens, separator)
                    if len(split_chunks) > 1:
                        made_progress = True
                        logger.debug(f"Split by '{separator}': {chunk_tokens} tokens → {len(split_chunks)} chunks")
                    new_chunks.extend(split_chunks)

            chunks = new_chunks

            # If all chunks are now within limits, we're done
            if all(self._count_tokens(chunk) <= max_tokens for chunk in chunks):
                logger.debug(f"Successfully split using separator '{separator}'")
                break

            # If we made no progress with this separator, continue to next
            if not made_progress:
                continue

        # Last resort: character-based splitting for any remaining oversized chunks
        final_chunks = []
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
            if chunk_tokens <= max_tokens:
                final_chunks.append(chunk)
            else:
                logger.warning(f"Using character-based splitting for {chunk_tokens}-token chunk")
                logger.info(f"this is the chunk: {chunk[:50]}... ({chunk_tokens} tokens)")

                # Estimate characters per token (rough approximation)
                chars_per_token = len(chunk) / max(1, chunk_tokens)
                estimated_max_chars = int(max_tokens * chars_per_token * 0.9)  # 10% safety margin

                # Character-based splitting as absolute last resort
                for i in range(0, len(chunk), estimated_max_chars):
                    sub_chunk = chunk[i:i + estimated_max_chars]
                    if sub_chunk.strip():
                        final_chunks.append(sub_chunk.strip())

        return final_chunks

    def _enforce_token_constraints(self, sentences: List[str]) -> List[str]:
        """
        Enforce token-based length constraints with hierarchical splitting.
        """
        if not sentences:
            return []

        result = []
        stats = {"too_short": 0, "too_long": 0, "split": 0, "kept": 0}

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            token_count = self._count_tokens(sentence)

            # Filter out sentences that are too short
            if self.min_sentence_tokens is not None and token_count < self.min_sentence_tokens:
                stats["too_short"] += 1
                continue

            # If sentence is too long, split it hierarchically
            if self.max_sentence_tokens is not None and token_count > self.max_sentence_tokens:
                stats["too_long"] += 1
                chunks = self._hierarchical_split(sentence, self.max_sentence_tokens)
                stats["split"] += len(chunks) - 1

                # Filter the resulting chunks
                for chunk in chunks:
                    chunk_tokens = self._count_tokens(chunk)
                    if chunk_tokens >= (self.min_sentence_tokens or 0):
                        result.append(chunk)
                        stats["kept"] += 1
            else:
                result.append(sentence)
                stats["kept"] += 1

        #if stats["too_short"] > 0 or stats["too_long"] > 0:
            #logger.info(f"Sentence filtering stats: {stats}")

        return result

    def _split_with_regex(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        if not text.strip():
            return []

        # First split on obvious sentence boundaries
        sentences = self.sentence_end_pattern.split(text)

        # Also try splitting on double newlines (paragraphs)
        all_sentences = []
        for sentence in sentences:
            if '\n\n' in sentence:
                all_sentences.extend(re.split(r'\n\s*\n', sentence))
            else:
                all_sentences.append(sentence)

        # Clean and filter sentences
        sentences = [s.strip() for s in all_sentences if s.strip()]

        # Apply token-based constraints with hierarchical splitting
        return self._enforce_token_constraints(sentences)

    def split_text(self, text: str) -> List[str]:
        """
        Split text into sentences with token-based length limits.
        """
        if not text or not text.strip():
            return []

        if self.use_model and self.model is not None:
            try:
                sentences = self.model.split(text, threshold=self.sentence_threshold)
                sentences = [s.strip() for s in sentences if s.strip()]
                return self._enforce_token_constraints(sentences)
            except Exception as e:
                logger.warning(f"Error using SaT model: {e}. Falling back to regex")
                return self._split_with_regex(text)
        else:
            return self._split_with_regex(text)

    def split_texts(self, texts: List[str]) -> List[List[str]]:
        """Split multiple texts into sentences."""
        if self.use_model and self.model is not None:
            try:
                sentence_lists = list(self.model.split(texts, threshold=self.sentence_threshold))
                sentence_lists = [[s.strip() for s in sentences if s.strip()] for sentences in sentence_lists]
                return [self._enforce_token_constraints(sentences) for sentences in sentence_lists]
            except Exception as e:
                logger.warning(f"Error using SaT model for batch splitting: {e}. Falling back to regex")
                return [self._split_with_regex(text) for text in texts]
        else:
            return [self._split_with_regex(text) for text in texts]

    def analyze_sentences(self, sentences: List[str]) -> dict:
        """Analyze sentence statistics for debugging."""
        if not sentences:
            return {"count": 0}

        token_counts = [self._count_tokens(s) for s in sentences]
        char_counts = [len(s) for s in sentences]

        stats = {
            "count": len(sentences),
            "token_stats": {
                "min": min(token_counts),
                "max": max(token_counts),
                "mean": sum(token_counts) / len(token_counts),
                "total": sum(token_counts)
            },
            "char_stats": {
                "min": min(char_counts),
                "max": max(char_counts),
                "mean": sum(char_counts) / len(char_counts),
                "total": sum(char_counts)
            },
            "avg_chars_per_token": sum(char_counts) / sum(token_counts) if sum(token_counts) > 0 else 0
        }

        return stats


def create_token_based_sentence_splitter(
    tokenizer: PreTrainedTokenizer,
    use_model: bool = True,
    model_name: str = "sat-3l",
    sentence_threshold: float = 0.2,
    device: Optional[Union[str, torch.device]] = None,
    max_sentence_tokens: Optional[int] = None,
    min_sentence_tokens: Optional[int] = None,
    fallback_separators: Optional[List[str]] = None,
    **kwargs
) -> TokenBasedSentenceSplitter:
    """
    Create a token-based sentence splitter.

    Args:
        tokenizer: The tokenizer to use for token counting
        max_sentence_tokens: Maximum tokens per sentence (not characters!)
        min_sentence_tokens: Minimum tokens per sentence (not characters!)
        Other args: Same as TokenBasedSentenceSplitter.__init__
    """
    return TokenBasedSentenceSplitter(
        tokenizer=tokenizer,
        use_model=use_model,
        model_name=model_name,
        sentence_threshold=sentence_threshold,
        device=device,
        max_sentence_tokens=max_sentence_tokens,
        min_sentence_tokens=min_sentence_tokens,
        fallback_separators=fallback_separators,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    # Test the fixed implementation
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    example_text = """This is a test sentence. This is another test sentence with more words to make it longer and see how the token-based splitting works. And here's a very long sentence that should definitely be split because it contains way too many tokens for our maximum limit, and we want to see how the hierarchical splitting approach works with commas, semicolons, and other punctuation marks to create reasonable splits."""

    # Create splitter with token limits
    splitter = create_token_based_sentence_splitter(
        tokenizer=tokenizer,
        use_model=False,  # Use regex for testing
        max_sentence_tokens=16,  # 16 tokens, not characters!
        min_sentence_tokens=2,   # 2 tokens, not characters!
    )

    print("Testing Token-Based Sentence Splitter")
    print("=" * 50)

    sentences = splitter.split_text(example_text)

    print(f"Split into {len(sentences)} sentences:")
    print()

    for i, sentence in enumerate(sentences):
        token_count = splitter._count_tokens(sentence)
        char_count = len(sentence)
        print(f"Sentence {i+1} ({token_count} tokens, {char_count} chars):")
        print(f"  {repr(sentence)}")
        print()

    # Show statistics
    stats = splitter.analyze_sentences(sentences)
    print("Statistics:")
    print(f"  Total sentences: {stats['count']}")
    print(f"  Token range: {stats['token_stats']['min']}-{stats['token_stats']['max']}")
    print(f"  Average tokens per sentence: {stats['token_stats']['mean']:.1f}")
    print(f"  Average characters per token: {stats['avg_chars_per_token']:.1f}")
