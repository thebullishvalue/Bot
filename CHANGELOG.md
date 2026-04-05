# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-04-05

### Added
- `regime.py` — 7-factor market regime detection with composite scoring (Momentum 30%, Trend 25%, Breadth 15%, Velocity 15%, Extremes 10%, Volatility 5%, Correlation 0%)
- `portfolio.py` — Conviction-based portfolio construction with style-aware dispersion weighting (SIP: +125%/-50%, Swing: +225%/-75%)
- `metrics.py` — Production-grade execution metrics with phase-level timing, error tracking, and performance benchmarks
- Conviction scoring system (0-100) using RSI (30%), Oscillator (30%), Z-Score (20%), and MA Alignment (20%)

### Changed
- Engine refactored from 4-phase pipeline to streamlined 2-phase pipeline (matches Pragyam 7.0.5 architecture)
- **Phase 1**: Data fetching + regime detection
- **Phase 2**: Conviction-based portfolio curation running ALL 95 strategies (no filtering)
- Replaced `curate_final_portfolio()` with `compute_conviction_based_weights()` for transparent conviction scoring
- Replaced `MarketRegimeDetectorV2` with `MarketRegimeDetector` from `regime.py`
- Updated README.md with new architecture documentation and file structure

### Removed
- `charts.py` — Zero traceable references across codebase
- `backtest_engine.py` — Imported but classes/functions never invoked
- `strategy_selection.py` — Imported but functions never used in conviction-based pipeline
- Dead imports in `engine.py` for removed modules
- Legacy `STRATEGY_SELECTION_AVAILABLE` and `DYNAMIC_SELECTION_AVAILABLE` flags (unused)

### Fixed
- Eliminated expensive walk-forward optimization and dynamic strategy selection bottlenecks
- Removed stale/unused code from engine import graph

## [4.0.0] - Previous Release

### Added
- Single entry point architecture (`app.py`)
- Thread-based Telegram bot with daemon lifecycle management
- WAL mode SQLite database for safe concurrent access
- Auto-writable path detection for cloud deployments
- Global error handler for graceful failure recovery
- Clean client handover with session claiming

---

*For earlier versions, see repository history.*
