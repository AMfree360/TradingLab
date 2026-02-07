# Trading Lab - Developer Documentation

Welcome to the developer documentation for Trading Lab! This documentation is designed for **developers and engineers** who want to extend, maintain, or contribute to the Trading Lab system.

## 📚 Documentation Index

### 🏗️ Architecture & Design
- **[Architecture Overview](01-architecture.md)** - System architecture, design patterns, and technical decisions
- **[Filter Abstraction Architecture](02-filter-abstraction.md)** - Strategy filter system design and implementation
- **[Validation Architecture](03-validation-architecture.md)** - Validation system design and philosophy
- **[Engine Layer Details](04-engine-layer.md)** - Backtest engine implementation details

### 🔧 Development Guides
- **[Strategy Development Guide](05-strategy-development.md)** - How to create and extend strategies (technical deep-dive)
- **[Extension Points](06-extension-points.md)** - How to extend the system (new strategies, metrics, validators, adapters)
- **[Development Guidelines](07-development-guidelines.md)** - Coding standards, best practices, and contribution guidelines

### 📊 Technical Deep-Dives
- **[Monte Carlo Implementation](08-monte-carlo.md)** - Monte Carlo validation tests implementation
- **[Leverage Architecture](09-leverage-architecture.md)** - Leverage and margin system design
- **[Market Agnostic Design](10-market-agnostic.md)** - How the system supports multiple markets

### 🔍 Reference
- **[API Reference](11-api-reference.md)** - Complete API documentation for all components
- **[Configuration Schema](12-configuration-schema.md)** - Configuration system internals

## 🎯 Quick Navigation

**Want to...**

- **Understand the system architecture?** → Start with [Architecture Overview](01-architecture.md)
- **Create a new strategy?** → See [Strategy Development Guide](05-strategy-development.md)
- **Add a new feature?** → Check [Extension Points](06-extension-points.md)
- **Understand validation?** → Read [Validation Architecture](03-validation-architecture.md)
- **Find API details?** → See [API Reference](11-api-reference.md)

## 🏛️ System Architecture Overview

Trading Lab follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│         User Interface Layer            │
│    (CLI Scripts, Configuration)        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Strategy Layer                  │
│    (StrategyBase, Custom Strategies)     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Engine Layer                    │
│    (BacktestEngine, Resampler)          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Validation Layer                │
│    (Walk-Forward, Monte Carlo)           │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Adapter Layer                    │
│    (Data Loaders, Exchange APIs)         │
└──────────────────────────────────────────┘
```

### Key Design Principles

1. **Strategy-Agnostic**: Engine doesn't know strategy details
2. **Modular**: Components are independent and replaceable
3. **Config-Driven**: Behavior controlled via YAML
4. **Reproducible**: All runs are deterministic
5. **Extensible**: Easy to add new components

## 🔑 Core Components

### Strategy Layer (`strategies/`)
- **StrategyBase**: Abstract base class for all strategies
- **Filter System**: Reusable filter components
- **Strategy Types**: Pre-configured strategy templates

### Engine Layer (`engine/`)
- **BacktestEngine**: Core backtesting execution
- **Resampler**: Timeframe conversion
- **Account**: Capital and position tracking
- **Broker**: Order execution simulation

### Validation Layer (`validation/`)
- **WalkForwardAnalyzer**: Out-of-sample validation
- **MonteCarloRunner**: Permutation and bootstrap tests
- **TrainingValidator**: Phase 1 validation
- **OOSValidator**: Phase 2 validation

### Metrics Layer (`metrics/`)
- **MetricsCalculator**: Performance metric calculations
- **Enhanced Metrics**: Sharpe, Sortino, CAGR, etc.

## 🛠️ Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd TradingLab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_backtest.py

# Run with coverage
pytest --cov=engine --cov=validation
```

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where possible
- Document all public APIs
- Write tests for new features

## 📝 Contributing

### Before Contributing

1. **Read the architecture docs** - Understand the system design
2. **Check existing issues** - See if your feature is already planned
3. **Discuss major changes** - Open an issue for discussion first

### Development Process

1. **Create a branch** from `main`
2. **Make changes** following development guidelines
3. **Write tests** for your changes
4. **Update documentation** if needed
5. **Submit pull request** with clear description

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance considered

## 🔍 Understanding the Codebase

### Key Directories

- `engine/` - Core backtesting engine
- `strategies/` - Strategy implementations
- `validation/` - Validation tests
- `metrics/` - Performance metrics
- `adapters/` - Data adapters
- `config/` - Configuration system
- `scripts/` - CLI tools
- `tests/` - Test suite

### Important Files

- `engine/backtest_engine.py` - Main backtest execution
- `strategies/base/strategy_base.py` - Strategy interface
- `validation/pipeline.py` - Validation pipeline
- `config/schema.py` - Configuration schema

## 🧪 Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock dependencies
- Test edge cases

### Integration Tests
- Test component interactions
- Use real data (small datasets)
- Verify end-to-end workflows

### Validation Tests
- Test validation logic
- Verify statistical correctness
- Check edge cases

## 📚 Additional Resources

- **User Documentation**: See `/docs/user/` for user-focused guides
- **Example Strategies**: Check `strategies/` for implementation examples
- **Configuration Examples**: See `config/` for configuration templates

## 🆘 Getting Help

If you're stuck:

1. **Read the relevant architecture doc** - Most questions are answered there
2. **Check the code** - Well-documented code is the best reference
3. **Look at examples** - See how similar features are implemented
4. **Ask questions** - Open an issue for discussion

## 🎓 Learning Path

**Week 1: Understanding the System**
- Read Architecture Overview
- Explore codebase structure
- Run example strategies

**Week 2: Core Components**
- Study BacktestEngine
- Understand StrategyBase
- Review validation system

**Week 3: Extension**
- Read Extension Points
- Create a simple strategy
- Add a custom metric

**Week 4: Advanced**
- Study filter system
- Understand Monte Carlo implementation
- Review advanced features

---

**Remember**: Good code is readable, testable, and maintainable. Follow the existing patterns and document your changes!

Happy coding! 🚀
