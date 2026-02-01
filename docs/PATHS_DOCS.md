# Paths Module Documentation

**Module Path**: `api/src/core/paths.py`

**Last Updated**: 2026-02-01

## Overview

The paths module provides centralized path management utilities for the Qwen3 TTS API. It ensures consistent path resolution across different environments (development, production, Docker containers) and simplifies file system operations.

## Key Functions

### `get_project_root()`

Returns the absolute path to the project root directory.

**Usage**:
```python
from api.src.core.paths import get_project_root

root = get_project_root()
# Returns: Path('/home/user/Apps/Qwen3-TTS-FastAPI')
```

### `resolve_path(path, base_dir=None)`

Resolves a path to an absolute path. If the path is relative, it's resolved against the project root (or provided base directory).

**Parameters**:
- `path`: The path to resolve (string or Path)
- `base_dir`: Optional base directory for relative paths

**Returns**: Absolute Path object

**Examples**:
```python
from api.src.core.paths import resolve_path

# Relative path
abs_path = resolve_path("models/base")
# Returns: Path('/home/user/Apps/Qwen3-TTS-FastAPI/models/base')

# Absolute path (unchanged)
abs_path = resolve_path("/absolute/path")
# Returns: Path('/absolute/path')

# Custom base directory
abs_path = resolve_path("config.json", base_dir=Path("/etc/myapp"))
# Returns: Path('/etc/myapp/config.json')
```

### `get_model_path(model_name)`

Returns the path to a specific model directory.

**Supported Models**:
- `base` - Base model for voice cloning
- `custom-voice` - Custom voice model with predefined speakers
- `voice-design` - Voice design model
- `base-large` - Large base model (1.7B parameters)
- `custom-voice-large` - Large custom voice model (1.7B parameters)
- `tokenizer` - Speech tokenizer model

**Returns**: Path object or None if model doesn't exist

**Usage**:
```python
from api.src.core.paths import get_model_path

model_path = get_model_path("custom-voice")
if model_path:
    print(f"Model found at: {model_path}")
else:
    print("Model not found")
```

### `ensure_dir(path)`

Ensures a directory exists, creating it (and parent directories) if necessary.

**Usage**:
```python
from api.src.core.paths import ensure_dir

# Create output directory
output_dir = ensure_dir("outputs/generated_audio")
# Directory is created if it doesn't exist
```

### `get_relative_path(path, base_dir=None)`

Converts an absolute path to a relative path from the base directory.

**Usage**:
```python
from api.src.core.paths import get_relative_path, get_project_root

abs_path = get_project_root() / "models" / "base"
rel_path = get_relative_path(abs_path)
# Returns: Path('models/base')
```

## Integration with Config Module

The paths module is used by the config module to resolve model paths:

```python
# In config.py
from api.src.core.paths import resolve_path

class Settings(BaseSettings):
    base_model_path: str = "models/base"
    
    def get_model_path(self, model_name: str) -> str | None:
        # ... logic to select path ...
        return str(resolve_path(path))
```

## Design Decisions

### 1. Path Objects Over Strings

All functions return `pathlib.Path` objects rather than strings because:
- Type safety and IDE support
- Cross-platform compatibility (Windows/Unix)
- Rich path manipulation methods
- Better handling of special characters

### 2. Automatic Path Resolution

Relative paths are automatically resolved to the project root:
- No need to manually construct absolute paths
- Works regardless of where the script is run from
- Consistent behavior across development and production

### 3. Existence Checking

`get_model_path()` checks if the directory exists:
- Prevents errors from using non-existent paths
- Returns None gracefully for missing models
- Allows fallback logic in calling code

## Common Patterns

### Pattern 1: Loading a Model

```python
from api.src.core.paths import get_model_path
from api.src.inference.qwen3_tts_model import Qwen3TTSModel

model_path = get_model_path("custom-voice")
if not model_path:
    raise FileNotFoundError("Custom voice model not found")

model = Qwen3TTSModel.from_pretrained(str(model_path))
```

### Pattern 2: Creating Output Directories

```python
from api.src.core.paths import ensure_dir
import uuid

output_dir = ensure_dir("outputs/audio")
filename = f"{uuid.uuid4()}.mp3"
output_path = output_dir / filename
```

### Pattern 3: Resolving Configuration Files

```python
from api.src.core.paths import resolve_path
import json

config_path = resolve_path("config/production.json")
with open(config_path) as f:
    config = json.load(f)
```

## Testing

When writing tests, you may want to mock the project root:

```python
import pytest
from unittest.mock import patch
from api.src.core.paths import get_project_root

def test_resolve_path():
    with patch('api.src.core.paths.get_project_root') as mock_root:
        mock_root.return_value = Path("/tmp/test_project")
        
        result = resolve_path("models/base")
        assert result == Path("/tmp/test_project/models/base")
```

## Future Enhancements

1. **Path Caching**: Cache resolved paths to avoid repeated I/O
2. **Environment-Specific Roots**: Support different roots for dev/staging/prod
3. **Path Validation**: Add validation for expected file types
4. **Path Templates**: Support template variables in paths (e.g., `{model_version}`)
