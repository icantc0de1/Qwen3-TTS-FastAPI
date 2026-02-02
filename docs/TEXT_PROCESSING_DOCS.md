# Text Processing Module Documentation

**Module Path**: `api/src/services/text_processing/`

**Last Updated**: 2026-02-01

## Overview

The text processing module provides intelligent text normalization for TTS (Text-to-Speech) generation. It transforms raw input text into a format optimized for speech synthesis by handling URLs, emails, phone numbers, units, money, and special characters.

**Attribution**: This text normalization system is derived from the excellent work by [remsky](https://github.com/remsky) in the [kokoro-fastapi](https://github.com/remsky/kokoro-fastapi) project, licensed under Apache 2.0.

## Module Structure

```
api/src/services/text_processing/
├── __init__.py          # Package exports
└── normalizer.py        # Text normalization implementation
```

## Key Components

### 1. NormalizationOptions (Pydantic Model)

**Purpose**: Configures which text normalization features to enable.

**Location**: `api/src/structures/schemas.py`

**Key Features**:
- Granular control over normalization features
- All features enabled by default for best TTS results
- Can be disabled entirely or selectively

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| normalize | bool | true | Master switch - enables/disables all normalization |
| unit_normalization | bool | false | Converts units (10KB → "10 kilobytes") |
| url_normalization | bool | true | Converts URLs to spoken form |
| email_normalization | bool | true | Converts emails to spoken form |
| optional_pluralization_normalization | bool | true | Handles optional plurals like "word(s)" |
| phone_normalization | bool | true | Converts phone numbers to spoken digits |
| replace_remaining_symbols | bool | true | Replaces special symbols with words |

**Usage Example**:
```python
from api.src.structures.schemas import NormalizationOptions

# Default options (all features enabled)
options = NormalizationOptions()

# Custom options - disable URL normalization
options = NormalizationOptions(
    normalize=True,
    url_normalization=False,  # Keep URLs as-is
    phone_normalization=True
)

# Disable all normalization
options = NormalizationOptions(normalize=False)
```

### 2. normalize_text() Function

**Purpose**: Main entry point for text normalization.

**Location**: `api/src/services/text_processing/normalizer.py`

**Signature**:
```python
def normalize_text(text: str, normalization_options: NormalizationOptions) -> str
```

**Parameters**:
- `text`: Raw input text to normalize
- `normalization_options`: Configuration specifying which features to apply

**Returns**: Normalized text ready for TTS processing

**Processing Order**:
1. Email normalization (if enabled)
2. URL normalization (if enabled)
3. Unit normalization (if enabled)
4. Optional pluralization handling (if enabled)
5. Phone number normalization (if enabled)
6. Quote and bracket normalization
7. CJK punctuation handling
8. Time format normalization
9. Money and number normalization
10. Symbol replacement (if enabled)
11. Whitespace cleanup

**Usage Example**:
```python
from api.src.services.text_processing import normalize_text
from api.src.structures.schemas import NormalizationOptions

options = NormalizationOptions()
text = "Visit https://example.com or call 555-123-4567"
result = normalize_text(text, options)
# Result: "Visit https example dot com or call five five five one two three four five six seven"
```

## Normalization Features

### URL Normalization

**Pattern**: URLs like `https://example.com/path`

**Conversion**: 
- `https://` → "https"
- `.` → "dot"
- `/` → "slash"
- Special characters expanded (e.g., `&` → "ampersand", `%` → "percent")

**Example**:
```
Input:  "Visit https://example.com/page"
Output: "Visit https example dot com slash page"
```

### Email Normalization

**Pattern**: Email addresses like `user@example.com`

**Conversion**:
- `@` → "at"
- `.` → "dot"
- Special characters expanded

**Example**:
```
Input:  "Contact user@example.com"
Output: "Contact user at example dot com"
```

### Phone Number Normalization

**Pattern**: Phone numbers in various formats

**Conversion**: Converts to spoken digit groups with proper pauses

**Example**:
```
Input:  "Call 555-123-4567"
Output: "Call five five five one two three four five six seven"

Input:  "Call +1 (555) 123-4567"
Output: "Call one five five five one two three four five six seven"
```

### Unit Normalization

**Pattern**: Measurements with units (10KB, 5m, 100mph)

**Supported Units**:
- Length: m, cm, mm, km, in, ft, yd, mi
- Mass: g, kg, mg
- Time: s, ms, min, h
- Volume: l, ml, cl, dl
- Speed: kph, mph, m/s
- Temperature: °c, °f, k
- Data: b, kb, mb, gb, tb, pb
- And more...

**Example**:
```
Input:  "File size is 10KB"
Output: "File size is 10 kilobit"

Input:  "Speed is 60mph"
Output: "Speed is 60 mile per hour"
```

### Money Normalization

**Pattern**: Currency amounts ($10.50, £100, €50)

**Conversion**: Converts to spoken form with currency name

**Example**:
```
Input:  "Price is $19.99"
Output: "Price is nineteen dollars and ninety nine cents"

Input:  "It costs £50"
Output: "It costs fifty pounds"
```

### Number Normalization

**Pattern**: Numeric values

**Conversion**: Converts to spoken words using `inflect` library

**Example**:
```
Input:  "There are 42 apples"
Output: "There are forty two apples"

Input:  "Temperature is 98.6"
Output: "Temperature is ninety eight point six"
```

### Time Normalization

**Pattern**: Time formats (HH:MM, HH:MM:SS with AM/PM)

**Conversion**: Converts to spoken time

**Example**:
```
Input:  "Meeting at 14:30"
Output: "Meeting at fourteen thirty"

Input:  "Start at 9:00 AM"
Output: "Start at nine o'clock AM"

Input:  "Duration 1:30:45"
Output: "Duration one thirty and forty five seconds"
```

### Symbol Replacement

**Pattern**: Special symbols and characters

**Conversions**:
- `~` → space
- `@` → "at"
- `#` → "number"
- `$` → "dollar"
- `%` → "percent"
- `&` → "and"
- `*` → space
- `_` → space
- `=` → "equals"
- `/` → "slash"
- `\` → space
- `|` → space

**Example**:
```
Input:  "Cost is $50 @ store"
Output: "Cost is fifty dollars at store"
```

## Text Segmentation

The text processing module provides utilities for splitting text into segments for streaming TTS generation.

### 1. split_into_sentences()

**Purpose**: Split text into sentences for streaming TTS generation.

**Features**:
- Smart sentence boundary detection with regex
- Handles common abbreviations (Mr., Dr., Mrs., etc.)
- Processes decimal numbers (3.14) correctly
- Converts CJK punctuation to standard sentence endings
- Combines very short sentences with next sentence
- Enforces maximum sentence length to prevent overly long chunks

**Signature**:
```python
def split_into_sentences(
    text: str, 
    max_length: int = 300, 
    min_length: int = 10
) -> List[str]
```

**Parameters**:
- `text`: Input text to split
- `max_length`: Maximum characters per sentence (default: 300)
- `min_length`: Minimum characters per sentence (default: 10)

**Examples**:
```python
from api.src.services.text_processing import split_into_sentences

# Basic sentence splitting
text = "Hello world! How are you? I'm fine."
sentences = split_into_sentences(text)
# Result: ['Hello world!', 'How are you?', "I'm fine."]

# With abbreviations (preserves Dr., Mr., etc.)
text = "Dr. Smith arrived. He was late."
sentences = split_into_sentences(text)
# Result: ['Dr. Smith arrived.', 'He was late.']

# CJK punctuation support
text = "你好世界！你好吗？我很好。"
sentences = split_into_sentences(text)
# Result: ['你好世界!', '你好吗?', '我很好.']

# Length control
text = "This is a very long sentence that exceeds the maximum length limit and should be split appropriately."
sentences = split_into_sentences(text, max_length=50)
# Result: ['This is a very long sentence that', 'exceeds the maximum length limit and', 'should be split appropriately.']
```

### 2. split_into_chunks()

**Purpose**: Split text into overlapping chunks for streaming TTS.

**Features**:
- Fixed-size chunking with overlap for smooth transitions
- Configurable chunk size and overlap amount
- Minimum chunk length enforcement
- Smart word boundary detection to avoid mid-word splits

**Signature**:
```python
def split_into_chunks(
    text: str,
    max_chars: int = 200,
    overlap: int = 20,
    min_chunk_len: int = 30,
) -> List[str]
```

**Parameters**:
- `text`: Input text to split
- `max_chars`: Maximum characters per chunk (default: 200)
- `overlap`: Number of characters to overlap between chunks (default: 20)
- `min_chunk_len`: Minimum chunk length (default: 30)

**Examples**:
```python
from api.src.services.text_processing import split_into_chunks

# Basic chunking
text = "This is a very long sentence that needs to be split into smaller chunks for streaming."
chunks = split_into_chunks(text, max_chars=150, overlap=15)
# Result: [
#   'This is a very long sentence that needs to be split into smaller',
#   'split into smaller chunks for streaming.'
# ]

# Very long text
text = "Chunk one. " * 20  # 20 repetitions
chunks = split_into_chunks(text, max_chars=100, overlap=10)
# Multiple overlapping chunks of ~100 characters each

# Edge cases
short_text = "Short text only."
chunks = split_into_chunks(short_text)
# Result: ['Short text only.']
```

### Usage in Streaming TTS

These segmentation functions are automatically used by `Qwen3TTSService` based on the `streaming_mode` parameter:

- `StreamingMode.SENTENCE`: Uses `split_into_sentences()`
- `StreamingMode.CHUNK`: Uses `split_into_chunks()`
- `StreamingMode.FULL`: No segmentation, processes entire text

**Integration Flow**:
```
Text Input → Normalization → Segmentation (based on streaming_mode) → TTS Generation → Audio Chunks
```

## Integration with TTS Service

### Automatic Normalization

The text normalization is automatically applied in `Qwen3TTSService.generate_speech()` before sending text to the TTS model.

**Flow**:
```
API Request → OpenAISpeechRequest → _normalize_input_text() → TTS Backend → Audio
```

**Service Method**:
```python
def _normalize_input_text(
    self, 
    text: str, 
    normalization_options: Optional[NormalizationOptions] = None
) -> str:
    """Normalize input text for better TTS pronunciation."""
    if normalization_options is None:
        normalization_options = NormalizationOptions()
    
    if not normalization_options.normalize:
        return text
    
    try:
        normalized = normalize_text(text, normalization_options)
        return normalized
    except Exception as e:
        logger.warning(f"Text normalization failed: {e}")
        return text  # Fallback to original text
```

### API Usage

Clients can control normalization via the API:

```json
{
  "model": "tts-1",
  "input": "Visit https://example.com or pay $50",
  "voice": "alloy",
  "normalization_options": {
    "normalize": true,
    "url_normalization": true,
    "unit_normalization": false
  }
}
```

If `normalization_options` is not provided, default settings are used (all features enabled).

## Configuration

### Environment Variables

No specific environment variables for text processing. Configure via API request or modify defaults in `NormalizationOptions` class.

### Default Behavior

By default, all normalization features are enabled:
- URLs are converted to spoken form
- Emails are expanded
- Phone numbers are spoken digit-by-digit
- Numbers are converted to words
- Money amounts are spoken
- Units are expanded
- Special symbols are replaced

### Disabling Normalization

To completely disable normalization:

**Option 1 - API Request**:
```json
{
  "input": "Your text here",
  "normalization_options": {
    "normalize": false
  }
}
```

**Option 2 - Selective Disabling**:
```json
{
  "input": "Visit https://example.com",
  "normalization_options": {
    "normalize": true,
    "url_normalization": false  
  }
}
```

## Best Practices

### 1. Keep Normalization Enabled

Normalization significantly improves TTS quality for:
- Technical documentation (URLs, code)
- Business content (prices, phone numbers)
- Scientific text (measurements, units)
- Mixed content (emails, addresses)

### 2. Selective Disabling

Disable specific features only when:
- URLs need to remain clickable in transcriptions
- Phone numbers should display in original format
- Code snippets must preserve exact syntax

### 3. Testing

Test normalization with your specific content:
```python
from api.src.services.text_processing import normalize_text
from api.src.structures.schemas import NormalizationOptions

test_cases = [
    "Visit https://example.com",
    "Call 555-123-4567",
    "Price: $19.99",
    "Size: 10KB"
]

options = NormalizationOptions()
for text in test_cases:
    result = normalize_text(text, options)
    print(f"Input:  {text}")
    print(f"Output: {result}\n")
```

### 4. Custom Vocabulary

For domain-specific terms that need special handling, consider pre-processing text before sending to the API.

## Performance Considerations

### Regex Compilation

All regex patterns are pre-compiled at module load time for optimal performance:
```python
EMAIL_PATTERN = re.compile(r"...", re.IGNORECASE)
URL_PATTERN = re.compile(r"...", re.IGNORECASE)
# etc.
```

### Processing Time

Normalization adds minimal overhead (~1-5ms per request) due to:
- Pre-compiled regex patterns
- Efficient string operations
- Early exit if normalization disabled

### Memory

The normalizer maintains minimal state:
- Pre-compiled regex patterns (shared across requests)
- Unit and TLD lookup dictionaries
- No per-request memory allocation beyond input/output strings

## Error Handling

The normalization system is designed to be resilient:

1. **Graceful Degradation**: If normalization fails, original text is returned
2. **Partial Processing**: Each feature is independent - one failure doesn't affect others
3. **Logging**: Failures are logged at WARNING level with context
4. **No Exceptions**: Normalization never raises exceptions to the caller

**Error Flow**:
```python
try:
    normalized = normalize_text(text, options)
except Exception as e:
    logger.warning(f"Text normalization failed: {e}")
    return text  # Return original on failure
```

## Testing

### Unit Tests

Test individual normalization features:
```python
def test_url_normalization():
    text = "Visit https://example.com"
    result = handle_url(re.search(URL_PATTERN, text))
    assert "example dot com" in result

def test_phone_normalization():
    text = "555-123-4567"
    result = handle_phone_number(re.search(PHONE_PATTERN, text))
    assert "five" in result
```

### Integration Tests

Test full pipeline:
```python
def test_service_integration():
    service = Qwen3TTSService(backend)
    text = "Visit https://example.com"
    result = service._normalize_input_text(text)
    assert "example dot com" in result
```

## Future Enhancements

1. **Additional Unit Types**: Support for more specialized units
2. **Multi-language Support**: Extend normalization for non-English text
3. **Custom Replacements**: Allow user-defined symbol mappings
4. **Context Awareness**: Smarter handling based on surrounding text
5. **Abbreviation Expansion**: Expand common abbreviations (Dr., Mr., etc.)

## References

- [kokoro-fastapi](https://github.com/remsky/kokoro-fastapi) - Original text normalization implementation
- [inflect](https://pypi.org/project/inflect/) - Number-to-words conversion library
- [OpenAI TTS API](https://platform.openai.com/docs/guides/text-to-speech) - API specification

## Integration Points

### Service Layer
```python
# api/src/services/qwen3_tts_service.py
from api.src.services.text_processing import normalize_text
from api.src.structures.schemas import NormalizationOptions

class Qwen3TTSService:
    def _normalize_input_text(self, text, options):
        return normalize_text(text, options or NormalizationOptions())
```

### Router Layer
```python
# api/src/routers/openai_compatible.py
# Normalization options passed via OpenAISpeechRequest
request.normalization_options  # Optional[NormalizationOptions]
```

### Schema Layer
```python
# api/src/structures/schemas.py
class NormalizationOptions(BaseModel):
    # Configuration schema
    pass
```
