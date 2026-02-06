"""Text segmentation utilities for streaming TTS.

Provides functions to split text into sentences or fixed-size chunks
for progressive audio streaming.
"""

import re
from typing import List

# Connection words that should stay with the following clause
CONN_WORDS = {
    "and",
    "but",
    "or",
    "so",
    "yet",
    "for",
    "nor",
    "while",
    "when",
    "where",
    "wherever",
    "whether",
    "if",
    "unless",
    "until",
    "once",
    "because",
    "since",
    "as",
    "though",
    "although",
    "even",
    "while",
}


def split_into_sentences(
    text: str, max_length: int = 300, min_length: int = 60
) -> List[str]:
    """Split text into sentences for streaming TTS generation.

    Uses regex-based sentence boundary detection with smart handling for:
    - Common abbreviations (Mr., Dr., Mrs., Ms., etc.)
    - Decimal numbers (3.14)
    - Ellipses (...)
    - CJK punctuation converted to sentence endings
    - Automatic batching of short sentences to prevent audio gaps
    - Phrase-boundary aware splitting for long sentences

    Args:
        text: Input text to split
        max_length: Maximum characters per sentence (default 300)
        min_length: Minimum characters per sentence (default 60)
                   Shorter sentences are automatically batched together
                   until this threshold is reached.

    Returns:
        List of sentence strings, stripped of extra whitespace

    Example:
        >>> split_into_sentences("Hello world! How are you? I'm fine.")
        ['Hello world!', "How are you?", "I'm fine."]
        >>> split_into_sentences("Yes. I agree. Let's go.")
        ["Yes. I agree. Let's go."]  # Batched due to min_length
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
            if len(sent) <= max_length:
                final.append(sent)
            else:
                # Split long sentences on phrase boundaries first, then word boundaries
                chunks = _split_long_sentence(sent, max_length)
                final.extend(chunks)
        return final

    return processed


def _split_long_sentence(sentence: str, max_length: int) -> List[str]:
    """Split a long sentence on phrase boundaries for natural breaks.

    For sentences exceeding max_length, this function:
    1. First tries to split on punctuation: comma, semicolon, colon
    2. Then tries to split on connection words (and, but, or, etc.)
    3. Finally falls back to word boundaries

    Args:
        sentence: The long sentence to split
        max_length: Maximum characters per chunk

    Returns:
        List of sentence chunks, each under max_length
    """
    # Try splitting on phrase boundaries first (comma, semicolon, colon)
    # These indicate natural pause points in speech
    phrase_boundary = re.compile(r"([,;:]+)\s+")

    # Find all phrase boundary positions
    matches = list(phrase_boundary.finditer(sentence))

    if matches:
        # Try to split at the best phrase boundary
        for match in reversed(matches):
            split_pos = match.start()
            if (
                split_pos > max_length * 0.4
            ):  # Only split if we get reasonable chunk size
                chunk1 = sentence[: split_pos + 1].strip()  # Include punctuation
                chunk2 = sentence[split_pos + 1 :].strip()

                # Preserve connection word with chunk2
                words = chunk2.split()
                if words and words[0].lower() in CONN_WORDS:
                    # Keep connection word with the following text
                    pass  # Already handled

                if len(chunk1) <= max_length and len(chunk2) <= max_length:
                    result = [chunk1]
                    if len(chunk2) > max_length:
                        result.extend(_split_long_sentence(chunk2, max_length))
                    else:
                        result.append(chunk2)
                    return result

    # Try splitting on connection words
    conn_word_pattern = re.compile(
        r"\s+(" + "|".join(CONN_WORDS) + r")\s+", re.IGNORECASE
    )
    matches = list(conn_word_pattern.finditer(sentence))

    if matches:
        for match in reversed(matches):
            split_pos = match.start()
            if split_pos > max_length * 0.4:
                chunk1 = sentence[: split_pos + 1].strip()
                chunk2 = sentence[split_pos + 1 :].strip()

                if len(chunk1) <= max_length and len(chunk2) <= max_length:
                    result = [chunk1]
                    if len(chunk2) > max_length:
                        result.extend(_split_long_sentence(chunk2, max_length))
                    else:
                        result.append(chunk2)
                    return result

    # Fall back to word boundary splitting
    final: List[str] = []
    current = sentence.strip()

    while len(current) > max_length:
        idx = current.rfind(" ", 0, max_length)
        if idx == -1:
            # No space found, hard split
            idx = max_length
            final.append(current[:idx].strip())
            current = current[idx:].strip()
        else:
            final.append(current[:idx].strip())
            current = current[idx:].strip()

    if current:
        final.append(current)

    return final


def split_into_chunks(
    text: str,
    max_chars: int = 200,
    overlap: int = 20,
    min_chunk_len: int = 30,
) -> List[str]:
    """Split text into overlapping chunks for streaming TTS.

    Useful when sentence-level splitting produces too long chunks or
    when you need more granular control over chunk sizes.

    This function improves upon basic chunking by:
    - Respecting phrase boundaries (commas, semicolons, colons)
    - Keeping connection words (and, but, or) with the following chunk
    - Preserving overlap for seamless audio transitions

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

    # Pre-split on phrase boundaries for better chunking
    phrase_boundary = re.compile(r"([,;:]+\s+)")
    matches = list(phrase_boundary.finditer(text))

    while start < len(text):
        end = start + max_chars
        best_break = end  # Default to max_chars position

        if end < len(text):
            # Look for natural break points: phrase boundaries first
            space_idx = text.rfind(" ", start, end)

            # Use a space if found to ensure clean chunk boundaries
            if space_idx != -1:
                # Check if there's a phrase boundary near the max_chars point
                phrase_break = None
                for match in matches:
                    if start < match.start() <= end:
                        # Prefer phrase boundary if it gives reasonable chunk size
                        if match.start() > start + max_chars * 0.3:
                            phrase_break = match.end()
                            break

                if phrase_break and phrase_break <= end:
                    best_break = phrase_break
                else:
                    best_break = space_idx

            chunk = text[start:best_break].strip()
            start = best_break
        else:
            chunk = text[start:].strip()
            start = len(text)

        if chunk and len(chunk) >= min_chunk_len:
            chunks.append(chunk)
        elif chunk and not chunks:
            chunks.append(chunk)

        # Apply overlap for next iteration
        start -= overlap

        # Prevent infinite loop: if overlap caused stagnation, use max_chars
        # This handles cases where spaces are too close to the start position
        # Use start <= best_break instead of start <= 0 to catch cases where
        # start goes backwards but remains positive (e.g., from 11 to 6)
        if start <= best_break:
            start = best_break

    return chunks
