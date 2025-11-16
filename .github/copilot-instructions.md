# GitHub Copilot Instructions - Python Code Quality

This repository follows strict Python code quality standards inspired by ArjanCodes. Apply these rules to all code suggestions.

I need you to use the production setup to run /scripts/train.py, /scripts/predict.py and /scripts/evaluate_holdout.py for testing code snippets.

Allways run the code using chunked execution to avoid memory issues. Take into account this applies for training, prediction and evaluation.

Use UV for running and testing code snippets. Since it will manage dependencies and environment, ensure compatibility.
## Core Principles

- **Clarity over cleverness**: Write idiomatic, maintainable Python
- **Type safety**: Always use type hints with precise types
- **Defensive programming**: Use assertions for invariants and preconditions
- **Modern Python**: Leverage Python 3.10+ features (match/case, union types, dataclasses)

## Style Guidelines

### Functions & Classes

```python
# ✅ Good: Small, focused function with type hints
def calculate_revenue(
    buyer_probability: float,
    predicted_revenue: float
) -> float:
    """Calculate expected revenue from buyer probability and prediction."""
    assert 0.0 <= buyer_probability <= 1.0, "buyer_probability must be in [0, 1]"
    assert predicted_revenue >= 0.0, "predicted_revenue must be non-negative"
    return buyer_probability * predicted_revenue

# ❌ Bad: No types, no validation, unclear name
def calc(x, y):
    return x * y
```

### Type Hints

```python
from __future__ import annotations
from typing import Sequence, Mapping
from pathlib import Path

# ✅ Use precise types
def load_config(path: Path) -> Mapping[str, Any]:
    ...

def process_files(files: Sequence[Path]) -> list[dict[str, float]]:
    ...

# ❌ Avoid bare types
def load_config(path):  # No type hints
    ...

def process_files(files: list) -> list:  # Too vague
    ...
```

### Dataclasses

```python
from dataclasses import dataclass

# ✅ Use dataclasses for structured data
@dataclass
class ModelMetrics:
    rmsle: float
    rmse: float
    mae: float
    r2: float
    
    def __post_init__(self) -> None:
        assert self.rmsle >= 0.0, "RMSLE must be non-negative"
        assert self.rmse >= 0.0, "RMSE must be non-negative"

# ❌ Avoid unstructured dictionaries
def calculate_metrics(y_true, y_pred):
    return {
        'rmsle': ...,
        'rmse': ...,
    }
```

## Defensive Programming

### Assertions for Invariants

```python
# ✅ Use assertions for development-time checks
def train_model(X: pd.DataFrame, y: np.ndarray) -> Model:
    """Train model on features X and targets y."""
    assert len(X) == len(y), "X and y must have same length"
    assert len(X) > 0, "Cannot train on empty dataset"
    assert not X.isna().any().any(), "X must not contain NaN values"
    
    model = fit_model(X, y)
    
    assert model.is_fitted, "Model should be fitted after training"
    return model

# ❌ Don't use assert for user input validation
def load_user_file(path: str) -> pd.DataFrame:
    assert os.path.exists(path)  # Bad: use exceptions instead
    return pd.read_csv(path)

# ✅ Use exceptions for user input
def load_user_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)
```

### Preconditions & Postconditions

```python
def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Logarithmic Error."""
    # Preconditions
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    assert np.all(y_true >= 0), "y_true must be non-negative for RMSLE"
    assert np.all(y_pred >= 0), "y_pred must be non-negative for RMSLE"
    
    result = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    
    # Postcondition
    assert result >= 0.0, "RMSLE must be non-negative"
    return result
```

## Avoiding Common Pitfalls

### Float Comparisons

```python
import math

# ✅ Use math.isclose for float comparisons
def is_valid_probability(p: float) -> bool:
    return -1e-9 <= p <= 1.0 + 1e-9  # Allow small tolerance

def values_match(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9)

# ❌ Never use == for computed floats
if revenue == 0.0:  # Bad if revenue is computed
    ...
```

### Mutable Defaults

```python
# ✅ Use None as default, create mutable in body
def process_data(
    items: list[str],
    exclude: list[str] | None = None
) -> list[str]:
    if exclude is None:
        exclude = []
    return [item for item in items if item not in exclude]

# ❌ Never use mutable defaults
def process_data(items: list[str], exclude: list[str] = []):  # Bug!
    ...
```

### Loop Variable Closures

```python
# ✅ Bind loop variable as default argument
callbacks = [lambda x, i=i: x * i for i in range(10)]

# ❌ Late binding will cause all to use last value
callbacks = [lambda x: x * i for i in range(10)]  # Bug!

# ✅ Better: use named function
def make_multiplier(factor: int):
    return lambda x: x * factor

callbacks = [make_multiplier(i) for i in range(10)]
```

## Enums & Constants

```python
from enum import Enum, IntEnum, auto

# ✅ Use enums for related constants
class ModelStage(Enum):
    BUYER_CLASSIFICATION = auto()
    REVENUE_REGRESSION = auto()
    ENSEMBLE = auto()
    
    def display_name(self) -> str:
        return self.name.replace('_', ' ').title()

# ✅ Use IntEnum for numeric constants
class DataSplit(IntEnum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

# ❌ Avoid magic strings/numbers
stage = "buyer_classification"  # Bad
stage = ModelStage.BUYER_CLASSIFICATION  # Good
```

## Pattern Matching

```python
from typing import Any

# ✅ Use match/case for complex branching
def process_value(value: Any) -> str:
    match value:
        case int() as i if i > 0:
            return f"Positive integer: {i}"
        case float() as f:
            return f"Float: {f:.2f}"
        case str() as s:
            return f"String: {s}"
        case list() as lst if len(lst) > 0:
            return f"Non-empty list with {len(lst)} items"
        case _:
            return "Unknown type"

# ✅ Use __match_args__ for custom classes
@dataclass
class Point:
    __match_args__ = ("x", "y")
    x: float
    y: float

def describe_point(p: Point) -> str:
    match p:
        case Point(0, 0):
            return "Origin"
        case Point(x, 0):
            return f"On X-axis at {x}"
        case Point(0, y):
            return f"On Y-axis at {y}"
        case Point(x, y):
            return f"At ({x}, {y})"
```

## Error Handling

```python
# ✅ Use specific exceptions
class ModelNotTrainedError(Exception):
    """Raised when attempting to predict with untrained model."""
    pass

def predict(model: Model, X: pd.DataFrame) -> np.ndarray:
    if not model.is_fitted:
        raise ModelNotTrainedError("Model must be trained before prediction")
    return model.predict(X)

# ❌ Never use bare except
try:
    result = risky_operation()
except:  # Bad!
    pass

# ✅ Catch specific exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except KeyError as e:
    logger.error(f"Missing key: {e}")
    raise
```

## Code Organization

```python
# ✅ Organize imports: stdlib, third-party, local
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.base import BaseModel
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_rmsle

# ✅ Group related functionality
class DataValidator:
    """Validates data before model training."""
    
    @staticmethod
    def validate_features(X: pd.DataFrame) -> None:
        """Validate feature matrix."""
        assert len(X) > 0, "Feature matrix must not be empty"
        assert not X.isna().any().any(), "Features must not contain NaN"
    
    @staticmethod
    def validate_targets(y: np.ndarray) -> None:
        """Validate target array."""
        assert len(y) > 0, "Target array must not be empty"
        assert np.all(np.isfinite(y)), "Targets must be finite"

# ❌ Avoid god objects or utils.py dumping grounds
class Utils:  # Bad: too generic
    def do_everything(self):
        ...
```

## Testing Mindset

```python
import pytest

# ✅ Write focused tests with clear structure
def test_rmsle_calculation():
    """Test RMSLE calculates correct value for known inputs."""
    # Given
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    
    # When
    result = calculate_rmsle(y_true, y_pred)
    
    # Then
    assert math.isclose(result, 0.0, abs_tol=1e-9)

def test_rmsle_raises_on_negative_values():
    """Test RMSLE raises assertion error for negative inputs."""
    y_true = np.array([1.0, -1.0])
    y_pred = np.array([1.0, 1.0])
    
    with pytest.raises(AssertionError, match="non-negative"):
        calculate_rmsle(y_true, y_pred)
```

## When Generating Code

For every code suggestion:

1. **Add type hints** to all functions and methods
2. **Add assertions** for invariants and preconditions
3. **Use dataclasses** instead of plain dicts for structured data
4. **Prefer enums** over magic strings/numbers
5. **Handle errors explicitly** with specific exceptions
6. **Keep functions small** (< 30-40 lines)
7. **Avoid known pitfalls**: float equality, mutable defaults, late binding
8. **Follow PEP 8**: snake_case, clear names, organized imports

## Key Patterns to Apply

```python
# Pattern: Input validation with assertions
def process_data(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    assert len(data) > 0, "data must not be empty"
    assert 'required_field' in config, "config must contain 'required_field'"
    ...

# Pattern: Enum for constants
class MetricType(Enum):
    RMSLE = "rmsle"
    RMSE = "rmse"
    MAE = "mae"

# Pattern: Dataclass for structured data
@dataclass
class TrainingConfig:
    n_estimators: int
    learning_rate: float
    max_depth: int
    
    def __post_init__(self) -> None:
        assert self.n_estimators > 0, "n_estimators must be positive"
        assert 0.0 < self.learning_rate <= 1.0, "learning_rate must be in (0, 1]"

# Pattern: Early returns to reduce nesting
def process_item(item: dict[str, Any]) -> str | None:
    if not item:
        return None
    
    if 'id' not in item:
        logger.warning("Item missing id")
        return None
    
    return process_valid_item(item)
```



Apply these patterns consistently across all Python code suggestions.
