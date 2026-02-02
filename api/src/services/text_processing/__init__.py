"""Text processing pipeline."""

from .normalizer import normalize_text
from .segmentation import split_into_sentences, split_into_chunks


__all__ = [
    "normalize_text",
    "split_into_sentences",
    "split_into_chunks",
]
