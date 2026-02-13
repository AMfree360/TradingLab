# Trading Lab Documentation

[![CI](https://github.com/AMfree360/Dev/actions/workflows/ci.yml/badge.svg)](https://github.com/AMfree360/Dev/actions/workflows/ci.yml)

Welcome to the Trading Lab documentation! This documentation is organized into two main sections based on your needs.

## 📚 Documentation Structure

### 👤 User Documentation (`/docs/user/`)

**For users who want to use Trading Lab to develop, test, and trade strategies.**

Comprehensive guides written for beginners, assuming no prior knowledge of programming or trading systems. Step-by-step instructions with detailed explanations.

**[→ Go to User Documentation](user/README.md)**

**Includes:**
- Getting Started Guide
- Complete Workflow (Development → Validation → Live Trading)
- Data Management
- Strategy Development
- Configuration (including Trade Management architecture)
- Backtesting
- Validation
- Optimization
- Reports (including Edge Latency metric)
- Live Trading
- Scripts Reference

### 👨‍💻 Developer Documentation (`/docs/dev/`)

**For developers and engineers who want to extend, maintain, or contribute to Trading Lab.**

Technical deep-dives into architecture, design patterns, extension points, and implementation details.

**[→ Go to Developer Documentation](dev/README.md)**

**Includes:**
- Architecture Overview
- Filter Abstraction Architecture
- Trade Management Abstraction (NEW)
- Validation Architecture
- Engine Layer Details
- Strategy Development (Technical)
- Extension Points
- Development Guidelines
- Monte Carlo Implementation
- Leverage Architecture
- Market Agnostic Design
- API Reference

## 🚀 Quick Start

### New to Trading Lab?

**Start here**: [User Documentation → Getting Started](user/01-getting-started.md)

Then read (recommended): [User Documentation → Why TradingLab Exists](user/00-why-tradinglab.md)

Then follow: [User Documentation → Complete Workflow](user/02-complete-workflow.md) ⭐ **Recommended**

### Want to Extend the System?

**Start here**: [Developer Documentation → Architecture Overview](dev/01-architecture.md)

Then see: [Developer Documentation → Extension Points](dev/06-extension-points.md)

## 📖 Documentation Index

### User-Focused Guides

All user documentation is in `/docs/user/`:

- **[Why TradingLab Exists](user/00-why-tradinglab.md)** - Philosophy, who it’s for, and what it protects you from

1. **[Getting Started](user/01-getting-started.md)** - Installation and first steps
2. **[Complete Workflow](user/02-complete-workflow.md)** - Step-by-step process ⭐ **START HERE**
3. **[Data Management](user/03-data-management.md)** - Getting and managing data
4. **[Strategy Development](user/13-strategy-development.md)** - Creating a strategy (user-friendly)
5. **[Configuration](user/04-configuration.md)** - Configuring strategies
6. **[Backtesting](user/05-backtesting.md)** - Running backtests
7. **[Validation](user/06-validation.md)** - Validating strategies
8. **[Optimization](user/07-optimization.md)** - Optimizing parameters
9. **[Reports](user/08-reports.md)** - Understanding results
10. **[Live Trading](user/09-live-trading.md)** - Running strategies live
11. **[Scripts Reference](user/10-scripts-reference.md)** - All command-line tools
12. **[Golden Path: Split Policy + Config Profiles](user/11-split-policy-and-profiles.md)** - Standardized, reproducible workflow

### Developer-Focused Guides

All developer documentation is in `/docs/dev/`:

1. **[Architecture Overview](dev/01-architecture.md)** - System architecture and design
2. **[Filter Abstraction](dev/02-filter-abstraction.md)** - Filter system design
3. **[Validation Architecture](dev/03-validation-architecture.md)** - Validation system design
4. **[Engine Layer](dev/04-engine-layer.md)** - Backtest engine details
5. **[Strategy Development](dev/05-strategy-development.md)** - Technical strategy guide
6. **[Extension Points](dev/06-extension-points.md)** - How to extend the system
7. **[Development Guidelines](dev/07-development-guidelines.md)** - Coding standards
8. **[Monte Carlo](dev/08-monte-carlo.md)** - Monte Carlo implementation
9. **[Leverage Architecture](dev/09-leverage-architecture.md)** - Leverage system design
10. **[Market Agnostic](dev/10-market-agnostic.md)** - Multi-market support
11. **[API Reference](dev/11-api-reference.md)** - Complete API documentation

## 🎯 Which Documentation Should I Read?

### I'm a User (Trading Strategy Developer)

**Read**: [User Documentation](user/README.md)

- You want to develop and test trading strategies
- You're new to programming or trading systems
- You want step-by-step guides with detailed explanations
- You want to understand how to use the system

**Start with**: [Complete Workflow Guide](user/02-complete-workflow.md)

### I'm a Developer (System Extender)

**Read**: [Developer Documentation](dev/README.md)

- You want to extend or modify the system
- You want to understand the architecture
- You want to add new features (strategies, metrics, validators, etc.)
- You want to contribute to the codebase

**Start with**: [Architecture Overview](dev/01-architecture.md)

### I'm Both

**Read both!** Start with user documentation to understand how to use the system, then read developer documentation to understand how to extend it.

## 🔍 Finding What You Need

### By Use Case

**"I want to test a trading idea"**
→ [User: Complete Workflow](user/02-complete-workflow.md)

**"I want to create a new strategy"**
→ [User: Strategy Development](user/README.md) (see user docs)
→ [Dev: Strategy Development](dev/05-strategy-development.md) (technical details)

**"I want to add a new feature"**
→ [Dev: Extension Points](dev/06-extension-points.md)

**"I want to understand the system architecture"**
→ [Dev: Architecture Overview](dev/01-architecture.md)

**"I want to validate my strategy"**
→ [User: Validation Guide](user/06-validation.md)

**"I want to go live"**
→ [User: Live Trading Guide](user/09-live-trading.md)

### By Topic

**Strategy Development**:
- User: [Strategy Development](user/README.md) (see user docs)
- Dev: [Strategy Development](dev/05-strategy-development.md) (technical)

**Testing & Validation**:
- User: [Backtesting](user/05-backtesting.md), [Validation](user/06-validation.md)
- Dev: [Validation Architecture](dev/03-validation-architecture.md)

**System Architecture**:
- Dev: [Architecture Overview](dev/01-architecture.md), [Engine Layer](dev/04-engine-layer.md)

**Extending the System**:
- Dev: [Extension Points](dev/06-extension-points.md), [Development Guidelines](dev/07-development-guidelines.md)

## 💡 Tips for Using This Documentation

1. **Start with the right section**: User docs for using, dev docs for extending
2. **Follow the workflow**: User docs have a recommended workflow path
3. **Read examples**: Both sections include practical examples
4. **Take notes**: Document what works for you
5. **Ask questions**: Review relevant sections when stuck

## 🆘 Getting Help

If you're stuck:

1. **Check the relevant guide**: Most questions are answered in the docs
2. **Review examples**: Each guide has working examples
3. **Check troubleshooting sections**: Guides include common issues
4. **Start simple**: Begin with basic examples, then add complexity

## 📝 Documentation Philosophy

### User Documentation

- **Assumes no prior knowledge**: Explains everything from scratch
- **Step-by-step**: Detailed instructions for every action
- **Beginner-friendly**: Simple language, no jargon
- **Practical**: Focus on "how to use" not "how it works"

### Developer Documentation

- **Technical deep-dive**: Detailed implementation information
- **Architecture-focused**: Understand design decisions
- **Extension-oriented**: How to add new features
- **Code examples**: Real implementation patterns

## 🎓 Learning Paths

### User Learning Path

**Week 1**: Getting Started + Data Management
**Week 2**: Strategy Development + Backtesting
**Week 3**: Validation + Reports
**Week 4**: Optimization + Configuration
**Week 5+**: Live Trading (when ready)

### Developer Learning Path

**Week 1**: Architecture Overview + Codebase Exploration
**Week 2**: Core Components (Engine, Validation, Metrics)
**Week 3**: Extension Points + Development Guidelines
**Week 4**: Advanced Topics (Monte Carlo, Filters, etc.)

## 📚 Additional Resources

- **Main README**: [../README.md](../README.md) - Project overview
- **Changelog**: [../CHANGELOG.md](../CHANGELOG.md) - Version history
- **Example Strategies**: `../strategies/` - Working examples
- **Configuration Examples**: `../config/` - Config templates

## 🎯 Key Concepts

Before diving in, understand these concepts:

- **Backtesting**: Testing strategies on historical data
- **Validation**: Rigorous testing to ensure robustness
- **Walk-Forward Analysis**: Testing on unseen data
- **Monte Carlo**: Testing if results are better than random
- **Edge Latency**: Statistical metric for detecting positive edge
- **Trade Management**: Abstracted system for stops, trailing, partials, TPs
- **Overfitting**: When strategy only works on training data
- **Risk Management**: Controlling position size and losses

## Summary

This documentation is organized to serve two audiences:

- **Users**: Comprehensive guides for developing and trading strategies
- **Developers**: Technical documentation for extending the system

Choose the section that matches your needs, and follow the recommended learning paths. Both sections are designed to be thorough and assume you're starting from scratch.

**Happy trading and coding!** 🚀
