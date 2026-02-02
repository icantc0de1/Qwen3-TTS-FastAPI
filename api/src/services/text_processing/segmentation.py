"""Text segmentation utilities for streaming TTS.

Provides functions to split text into sentences or fixed-size chunks
for progressive audio streaming."""

import re
from typing import List


def split_into_sentences(
    text: str, max_length: int = 300, min_length: int = 10
) -> List[str]:
    """Split text into sentences for streaming TTS generation.

    Uses regex-based sentence boundary detection with smart handling for:
    - Common abbreviations (Mr., Dr., Mrs., Ms., etc.)
    - Decimal numbers (3.14)
    - Ellipses (...)
    - CJK punctuation converted to sentence endings

    Args:
        text: Input text to split
        max_length: Maximum characters per sentence (default 300)
        min_length: Minimum characters per sentence (default 10)
                   Shorter sentences are combined with next sentence

    Returns:
        List of sentence strings, stripped of extra whitespace

    Example:
        >>> split_into_sentences("Hello world! How are you? I'm fine.")
        ['Hello world!', "How are you?", "I'm fine."]
    """
    SENTENCE_ENDINGS = re.compile(r"[.!?]+\s*")
    ELLIPSIS = re.compile(r"\.\s*\.\s*\.\s*")
    DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
    CJK_PUNCT = re.compile(r"[。！？]+")

    text = CJK_PUNCT.sub(
        lambda m: ". " if m.group() in "。" else ("! " if m.group() in "！" else "? "),
        text,
    )

    text = ELLIPSIS.sub("... ", text)

    abbreviations = [
        "Mr.",
        "Mr ",
        "Mrs.",
        "Mrs ",
        "Ms.",
        "Ms ",
        "Dr.",
        "Dr ",
        "Prof.",
        "Prof ",
        "Rev.",
        "Rev ",
        "Hon.",
        "Hon ",
        "Sen.",
        "Sen ",
        "Rep.",
        "Rep ",
        "Gov.",
        "Gov ",
        "Gen.",
        "Gen ",
        "Col.",
        "Col ",
        "Maj.",
        "Maj ",
        "Capt.",
        "Capt ",
        "Lt.",
        "Lt ",
        "Sgt.",
        "Sgt ",
        " Pvt.",
        " Pvt ",
        "Corp.",
        "Corp ",
        "Ltd.",
        "Ltd ",
        "Inc.",
        "Inc ",
        "Co.",
        "Co ",
        "vs.",
        "vs ",
        "etc.",
        "etc ",
        "e.g.",
        "i.e.",
        "cf.",
        "viz.",
        "ex.",
        "No.",
        "Nos ",
        "St.",
        "St ",
        "Ave.",
        "Hwy.",
        "Rd.",
        "Blvd.",
    ]

    temp_text = text
    for abbrev in abbreviations:
        if abbrev.endswith("."):
            placeholder = abbrev[:-1].replace(".", "_ABBREV_")
            temp_text = temp_text.replace(abbrev, placeholder + ".")

    temp_text = DECIMAL.sub("_DECIMAL_", temp_text)

    temp_text = re.sub(r"\s+", " ", temp_text)

    raw_sentences = SENTENCE_ENDINGS.split(temp_text)

    sentences: List[str] = []
    current_buffer = ""

    for raw in raw_sentences:
        if not raw.strip():
            continue

        current = raw
        current = re.sub(r"_ABBREV_", ".", current)
        current = re.sub(r"_DECIMAL_", ".", current)

        current = current.strip()
        if not current:
            continue

        if current_buffer:
            current = current_buffer + " " + current
            current_buffer = ""

        if len(current) >= min_length or current in [".", "!", "?"]:
            sentences.append(current)
            current_buffer = ""
        elif len(current) < min_length:
            current_buffer = current

    if current_buffer:
        if sentences:
            sentences[-1] = sentences[-1] + " " + current_buffer
        else:
            sentences.append(current_buffer)

    processed: List[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue

        has_ending = any(
            s.rstrip().endswith(ending)
            for ending in [".", "!", "?", "...", "。", "！", "？"]
        )
        if not has_ending:
            s = s + "."

        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        if s:
            processed.append(s)

    if not processed and text.strip():
        processed = [text.strip()]

    if max_length > 0:
        final: List[str] = []
        for sent in processed:
            while len(sent) > max_length:
                idx = sent.rfind(" ", 0, max_length)
                if idx == -1:
                    idx = max_length
                final.append(sent[:idx])
                sent = sent[idx:].strip()
            final.append(sent)
        return final

    return processed


def split_into_chunks(
    text: str,
    max_chars: int = 200,
    overlap: int = 20,
    min_chunk_len: int = 30,
) -> List[str]:
    """Split text into overlapping chunks for streaming TTS.

    Useful when sentence-level splitting produces too long chunks or
    when you need more granular control over chunk sizes.

    Args:
        text: Input text to split
        max_chars: Maximum characters per chunk (default 200)
        overlap: Number of characters to overlap between chunks (default 20)
        min_chunk_len: Minimum chunk length (default 30)

    Returns:
        List of text chunks with overlap applied

    Example:
        >>> split_into_chunks("This is a very long sentence that needs to be split into smaller chunks for streaming.")
        ['This is a very long sentence that needs to be split into smaller chunks for',
        'chunks for streaming.']
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end < len(text):
            space_idx = text.rfind(" ", start, end)
            if space_idx != -1:
                chunk = text[start:space_idx].strip()
                start = space_idx
            else:
                chunk = text[start:end].strip()
                start = end
        else:
            chunk = text[start:].strip()
            start = len(text)

        if chunk and len(chunk) >= min_chunk_len:
            chunks.append(chunk)
        elif chunk and not chunks:
            chunks.append(chunk)

        start -= overlap

    return chunks
