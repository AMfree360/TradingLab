# Development Guidelines

This document outlines coding standards, best practices, and contribution guidelines for Trading Lab development.

## Table of Contents

1. [Code Style](#code-style)
2. [Naming Conventions](#naming-conventions)
3. [Documentation Standards](#documentation-standards)
4. [Testing Requirements](#testing-requirements)
5. [Git Workflow](#git-workflow)
6. [Performance Guidelines](#performance-guidelines)
7. [Security Considerations](#security-considerations)
8. [Error Handling](#error-handling)

## Code Style

### Python Style Guide

Follow **PEP 8** Python style guide with these additions:

#### Line Length
- Maximum 100 characters per line
- Break long lines at logical points
- Use parentheses for line continuation

#### Imports
- Group imports: standard library, third-party, local
- Use absolute imports
- One import per line for clarity

```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
import pandas as pd
import numpy as np

# Local
from engine.backtest_engine import BacktestEngine
from strategies.base import StrategyBase
```

#### Type Hints

Always use type hints for function signatures:

```python
def calculate_metric(
    result: BacktestResult,
    window: int = 20
) -> float:
    """Calculate metric."""
    # ...
```

#### Docstrings

Use Google-style docstrings:

```python
def process_data(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Process DataFrame with threshold filtering.
    
    Args:
        df: Input DataFrame with OHLCV data
        threshold: Minimum value threshold (default: 0.5)
    
    Returns:
        Processed DataFrame with filtered rows
    
    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    # ...
```

## Naming Conventions

### Variables and Functions

- **snake_case** for variables and functions
- **Descriptive names**: `calculate_profit_factor()` not `calc_pf()`
- **Boolean prefixes**: `is_`, `has_`, `should_`, `can_`

```python
# Good
def should_allow_trade(context: FilterContext) -> bool:
    is_profitable = result.total_pnl > 0
    has_sufficient_trades = len(result.trades) >= 30

# Bad
def allow(context):  # Missing type hints, unclear name
    profit = result.pnl > 0  # Unclear boolean meaning
```

### Classes

- **PascalCase** for class names
- **Descriptive names**: `BacktestEngine` not `Engine`

```python
# Good
class WalkForwardAnalyzer:
    pass

class MonteCarloPermutation:
    pass

# Bad
class WFA:  # Abbreviation unclear
    pass
```

### Constants

- **UPPER_SNAKE_CASE** for constants
- **Module-level constants**: Define at top of file

```python
# Good
DEFAULT_INITIAL_CAPITAL = 10000.0
MIN_TRADES_FOR_VALIDATION = 30
MAX_DRAWDOWN_THRESHOLD = 0.20

# Bad
defaultCapital = 10000.0  # Wrong case
```

### Private Methods/Attributes

- **Leading underscore** for private/internal
- **Double underscore** only for name mangling (rarely needed)

```python
class BacktestEngine:
    def __init__(self):
        self.account = Account()  # Public
        self._open_positions = []  # Private
        self._equity_list = []  # Private
    
    def run(self, data: pd.DataFrame) -> BacktestResult:
        """Public method."""
        return self._execute_backtest(data)
    
    def _execute_backtest(self, data: pd.DataFrame) -> BacktestResult:
        """Private method."""
        # ...
```

## Documentation Standards

### Function Documentation

Every public function must have:
- **Purpose**: What it does
- **Args**: All parameters with types and descriptions
- **Returns**: Return type and description
- **Raises**: Exceptions that may be raised
- **Examples**: For complex functions

```python
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio from return series.
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year for annualization (default: 252)
    
    Returns:
        Sharpe ratio (annualized)
    
    Raises:
        ValueError: If returns is empty or contains only NaN values
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, 0.01])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe: {sharpe:.2f}")
        Sharpe: 0.58
    """
    if len(returns) == 0:
        raise ValueError("Returns series is empty")
    # ...
```

### Class Documentation

Every class must have:
- **Purpose**: What the class represents
- **Attributes**: Key attributes (if significant)
- **Example**: Usage example

```python
class BacktestEngine:
    """
    Core backtesting engine for executing trading strategies.
    
    The engine manages position lifecycle, capital tracking, and trade execution.
    It works with any strategy that implements the StrategyBase interface.
    
    Attributes:
        account: AccountState object tracking capital and positions
        strategy: Strategy instance generating trading signals
        trades: List of completed Trade objects
    
    Example:
        >>> strategy = MyStrategy(config)
        >>> engine = BacktestEngine(strategy=strategy, initial_capital=10000.0)
        >>> result = engine.run(data)
        >>> print(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
    """
    # ...
```

### Module Documentation

Every module should have:
- **Purpose**: What the module provides
- **Key Classes/Functions**: Main exports

```python
"""
Backtesting engine module.

This module provides the core backtesting engine for executing trading strategies
on historical data. It handles position management, capital tracking, and trade
execution.

Key Classes:
    BacktestEngine: Main engine for running backtests
    BacktestResult: Results container with metrics and trades

Key Functions:
    run_backtest(): Convenience function for running backtests
"""
```

## Testing Requirements

### Test Coverage

- **Minimum 80% coverage** for new code
- **100% coverage** for critical paths (validation, risk management)
- **Test edge cases**: Empty data, boundary conditions, error cases

### Test Structure

```python
# tests/test_backtest_engine.py

import pytest
import pandas as pd
from engine.backtest_engine import BacktestEngine
from strategies.base import StrategyBase


class TestBacktestEngine:
    """Test suite for BacktestEngine."""
    
    def test_basic_backtest(self):
        """Test basic backtest execution."""
        # Arrange
        strategy = MockStrategy()
        engine = BacktestEngine(strategy=strategy, initial_capital=10000.0)
        data = create_test_data()
        
        # Act
        result = engine.run(data)
        
        # Assert
        assert result is not None
        assert len(result.trades) > 0
        assert result.metrics['total_pnl'] is not None
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises appropriate error."""
        # Arrange
        strategy = MockStrategy()
        engine = BacktestEngine(strategy=strategy)
        data = pd.DataFrame()
        
        # Act & Assert
        with pytest.raises(ValueError, match="Data cannot be empty"):
            engine.run(data)
    
    def test_insufficient_capital(self):
        """Test behavior with insufficient capital."""
        # ...
```

### Test Naming

- **Descriptive names**: `test_calculate_sharpe_with_zero_returns`
- **Test structure**: `test_<what>_<condition>_<expected_result>`
- **Group related tests**: Use test classes

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=engine --cov=validation --cov-report=html

# Run specific test
pytest tests/test_backtest_engine.py::TestBacktestEngine::test_basic_backtest

# Run with verbose output
pytest -v
```

## Git Workflow

### Branch Naming

- **Feature**: `feature/add-custom-metric`
- **Bugfix**: `bugfix/fix-drawdown-calculation`
- **Refactor**: `refactor/simplify-validation-pipeline`
- **Docs**: `docs/update-api-reference`

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `perf`: Performance improvements

**Examples**:

```
feat(validation): add binomial test to OOS validation

Adds binomial statistical test to verify that success rate
is significantly better than random chance.

Closes #123
```

```
fix(engine): correct drawdown calculation for zero equity

Fixes division by zero error when equity reaches zero.
Adds proper handling for edge case.

Fixes #456
```

### Pull Request Process

1. **Create feature branch** from `main`
2. **Make changes** following guidelines
3. **Write tests** for new code
4. **Update documentation** if needed
5. **Run tests** and ensure they pass
6. **Create PR** with clear description
7. **Address review comments**
8. **Merge** after approval

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Performance Guidelines

### Vectorization

Use vectorized operations instead of loops:

```python
# Good - Vectorized
df['sma'] = df['close'].rolling(window=20).mean()
df['returns'] = df['close'].pct_change()

# Bad - Loop
sma = []
for i in range(len(df)):
    window = df['close'].iloc[max(0, i-19):i+1]
    sma.append(window.mean())
```

### Caching

Cache expensive calculations:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param1: float, param2: float) -> float:
    """Expensive calculation that can be cached."""
    # ...
```

### Memory Efficiency

- **Use generators** for large datasets
- **Delete large objects** when done
- **Avoid unnecessary copies**

```python
# Good - Generator
def process_trades(trades: List[Trade]):
    for trade in trades:
        yield process_trade(trade)

# Bad - List comprehension creates full list
results = [process_trade(t) for t in trades]  # If trades is huge
```

## Security Considerations

### API Keys

- **Never commit** API keys or secrets
- **Use environment variables** or `.env` files
- **Add to `.gitignore`**

```python
# Good
import os
api_key = os.getenv('BINANCE_API_KEY')

# Bad
api_key = "your_secret_key_here"  # Never do this!
```

### Input Validation

Always validate user input:

```python
def run_backtest(
    strategy_name: str,
    data_path: str,
    initial_capital: float
) -> BacktestResult:
    """Run backtest with validation."""
    # Validate inputs
    if not strategy_name or not isinstance(strategy_name, str):
        raise ValueError("strategy_name must be non-empty string")
    
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # ...
```

### File Paths

Sanitize file paths to prevent directory traversal:

```python
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    """Load data with path validation."""
    # Resolve to absolute path
    resolved = Path(path).resolve()
    
    # Ensure it's within allowed directory
    allowed_dir = Path('/allowed/data/dir').resolve()
    if not str(resolved).startswith(str(allowed_dir)):
        raise ValueError("Path outside allowed directory")
    
    # ...
```

## Error Handling

### Exception Types

Use appropriate exception types:

```python
# ValueError for invalid values
if value < 0:
    raise ValueError("Value must be non-negative")

# TypeError for wrong types
if not isinstance(data, pd.DataFrame):
    raise TypeError("data must be pandas DataFrame")

# FileNotFoundError for missing files
if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")

# Custom exceptions for domain-specific errors
class InsufficientDataError(Exception):
    """Raised when data is insufficient for operation."""
    pass
```

### Error Messages

- **Be specific**: Include context
- **Be actionable**: Suggest fixes
- **Include values**: Show what went wrong

```python
# Good
if len(trades) < min_trades:
    raise ValueError(
        f"Insufficient trades for validation: {len(trades)} < {min_trades}. "
        f"Need at least {min_trades} trades. Consider using a longer data period."
    )

# Bad
if len(trades) < min_trades:
    raise ValueError("Not enough trades")  # Too vague
```

### Logging

Use logging for important events:

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data: pd.DataFrame):
    """Process data with logging."""
    logger.info(f"Processing {len(data)} rows")
    
    try:
        result = expensive_operation(data)
        logger.info(f"Successfully processed {len(result)} rows")
        return result
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        raise
```

## Code Review Checklist

When reviewing code, check:

- [ ] **Style**: Follows PEP 8 and project conventions
- [ ] **Type hints**: All functions have type hints
- [ ] **Documentation**: Docstrings for all public functions/classes
- [ ] **Tests**: Adequate test coverage
- [ ] **Error handling**: Proper exception handling
- [ ] **Performance**: No obvious performance issues
- [ ] **Security**: No security vulnerabilities
- [ ] **Edge cases**: Handles edge cases properly
- [ ] **Breaking changes**: Documented if any

## Resources

- **PEP 8**: Python style guide
- **Google Python Style Guide**: Additional conventions
- **Type Hints**: PEP 484, PEP 526
- **Docstrings**: Google style guide
- **Testing**: pytest documentation

---

**Remember**: Good code is readable, maintainable, and well-tested. When in doubt, prioritize clarity over cleverness!
