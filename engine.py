"""
PRAGYAM Headless Engine
━━━━━━━━━━━━━━━━━━━━━━━
Runs the full Pragyam 7.0.5 pipeline without Streamlit dependencies.
Used by the Telegram bot to generate portfolios programmatically.

Architecture (matches Pragyam 7.0.5 app.py):
  Phase 1: Data fetching + regime detection
  Phase 2: Conviction-based portfolio curation (ALL strategies)

Pipeline Flow:
  1. Fetch historical data for all symbols
  2. Detect market regime using 7-factor composite
  3. Run ALL 95+ strategies (no filtering)
  4. Aggregate all holdings from all strategies
  5. Compute conviction scores (RSI, OSC, Z-Score, MA)
  6. Apply conviction-based weighting with style dispersion
  7. Select top 30 positions by conviction score

Version: 7.0.5 (Bot Edition)
"""

import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger("pragyam.engine")

# ─── Import Pragyam modules ───
from backdata import (
    generate_historical_data,
    load_symbols_from_file,
    MAX_INDICATOR_PERIOD,
    SYMBOLS_UNIVERSE
)
from regime import (
    MarketRegimeDetector,
    REGIME_MIX_MAP,
    compute_conviction_signals
)
from portfolio import compute_conviction_based_weights
from metrics import get_metrics, reset_metrics

# ─── Import all strategy classes from strategies.py ───
from strategies import (
    BaseStrategy,
    PRStrategy, CL1Strategy, CL2Strategy, CL3Strategy,
    MOM1Strategy, MOM2Strategy, MomentumMasters, VolatilitySurfer,
    AdaptiveVolBreakout, VolReversalHarvester, AlphaSurge, ReturnPyramid,
    MomentumCascade, AlphaVortex, SurgeSentinel, VelocityVortex,
    BreakoutAlphaHunter, ExtremeMomentumBlitz, HyperAlphaIgniter,
    VelocityApocalypse, QuantumMomentumLeap, NebulaMomentumStorm,
    ResonanceEcho, DivergenceMirage, FractalWhisper, InterferenceWave,
    ShadowPuppet, EntangledMomentum, ButterflyChaos, SynapseFiring,
    HolographicMomentum, WormholeTemporal, SymbioticAlpha, PhononVibe,
    HorizonEvent, EscherLoop, MicrowaveCosmic, SingularityMomentum,
    MultiverseAlpha, EternalReturnCycle, DivineMomentumOracle,
    CelestialAlphaForge, InfiniteMomentumLoop, GodParticleSurge,
    NirvanaMomentumWave, PantheonAlphaRealm, ZenithMomentumPeak,
    OmniscienceReturn, ApotheosisMomentum, TranscendentAlpha,
    TurnaroundSniper, MomentumAccelerator, VolatilityRegimeTrader,
    CrossSectionalAlpha, DualMomentum, AdaptiveZScoreEngine,
    MomentumDecayModel, InformationRatioOptimizer,
    BayesianMomentumUpdater, RelativeStrengthRotator,
    VolatilityAdjustedValue, NonlinearMomentumBlender,
    EntropyWeightedSelector, KalmanFilterMomentum,
    MeanVarianceOptimizer, RegimeSwitchingStrategy,
    FractalMomentumStrategy, CopulaBlendStrategy,
    WaveletDenoiser, GradientBoostBlender,
    AttentionMechanism, EnsembleVotingStrategy,
    OptimalTransportBlender, StochasticDominance,
    MaximumEntropyStrategy, HiddenMarkovModel,
    QuantileRegressionStrategy, MutualInformationBlender,
    GameTheoreticStrategy, ReinforcementLearningInspired,
    SpectralClusteringStrategy, CausalInferenceStrategy,
    BootstrapConfidenceStrategy, KernelDensityStrategy,
    SurvivalAnalysisStrategy, PrincipalComponentStrategy,
    FactorMomentumStrategy, ElasticNetBlender,
    RobustRegressionStrategy, ConvexOptimizationStrategy,
    MonteCarloStrategy, VariationalInferenceStrategy,
    NeuralNetworkInspired, GraphNeuralInspired,
    ContrastiveLearningStrategy,
)

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def get_all_strategies() -> Dict[str, BaseStrategy]:
    """Instantiate all available strategies."""
    strat_classes = {
        'PRStrategy': PRStrategy, 'CL1Strategy': CL1Strategy, 'CL2Strategy': CL2Strategy,
        'CL3Strategy': CL3Strategy, 'MOM1Strategy': MOM1Strategy, 'MOM2Strategy': MOM2Strategy,
        'MomentumMasters': MomentumMasters, 'VolatilitySurfer': VolatilitySurfer,
        'AdaptiveVolBreakout': AdaptiveVolBreakout, 'VolReversalHarvester': VolReversalHarvester,
        'AlphaSurge': AlphaSurge, 'ReturnPyramid': ReturnPyramid, 'MomentumCascade': MomentumCascade,
        'AlphaVortex': AlphaVortex, 'SurgeSentinel': SurgeSentinel, 'VelocityVortex': VelocityVortex,
        'BreakoutAlphaHunter': BreakoutAlphaHunter, 'ExtremeMomentumBlitz': ExtremeMomentumBlitz,
        'HyperAlphaIgniter': HyperAlphaIgniter, 'VelocityApocalypse': VelocityApocalypse,
        'QuantumMomentumLeap': QuantumMomentumLeap, 'NebulaMomentumStorm': NebulaMomentumStorm,
        'ResonanceEcho': ResonanceEcho, 'DivergenceMirage': DivergenceMirage,
        'FractalWhisper': FractalWhisper, 'InterferenceWave': InterferenceWave,
        'ShadowPuppet': ShadowPuppet, 'EntangledMomentum': EntangledMomentum,
        'ButterflyChaos': ButterflyChaos, 'SynapseFiring': SynapseFiring,
        'HolographicMomentum': HolographicMomentum, 'WormholeTemporal': WormholeTemporal,
        'SymbioticAlpha': SymbioticAlpha, 'PhononVibe': PhononVibe,
        'HorizonEvent': HorizonEvent, 'EscherLoop': EscherLoop,
        'MicrowaveCosmic': MicrowaveCosmic, 'SingularityMomentum': SingularityMomentum,
        'MultiverseAlpha': MultiverseAlpha, 'EternalReturnCycle': EternalReturnCycle,
        'DivineMomentumOracle': DivineMomentumOracle, 'CelestialAlphaForge': CelestialAlphaForge,
        'InfiniteMomentumLoop': InfiniteMomentumLoop, 'GodParticleSurge': GodParticleSurge,
        'NirvanaMomentumWave': NirvanaMomentumWave, 'PantheonAlphaRealm': PantheonAlphaRealm,
        'ZenithMomentumPeak': ZenithMomentumPeak, 'OmniscienceReturn': OmniscienceReturn,
        'ApotheosisMomentum': ApotheosisMomentum, 'TranscendentAlpha': TranscendentAlpha,
        'TurnaroundSniper': TurnaroundSniper, 'MomentumAccelerator': MomentumAccelerator,
        'VolatilityRegimeTrader': VolatilityRegimeTrader, 'CrossSectionalAlpha': CrossSectionalAlpha,
        'DualMomentum': DualMomentum, 'AdaptiveZScoreEngine': AdaptiveZScoreEngine,
        'MomentumDecayModel': MomentumDecayModel, 'InformationRatioOptimizer': InformationRatioOptimizer,
        'BayesianMomentumUpdater': BayesianMomentumUpdater, 'RelativeStrengthRotator': RelativeStrengthRotator,
        'VolatilityAdjustedValue': VolatilityAdjustedValue, 'NonlinearMomentumBlender': NonlinearMomentumBlender,
        'EntropyWeightedSelector': EntropyWeightedSelector, 'KalmanFilterMomentum': KalmanFilterMomentum,
        'MeanVarianceOptimizer': MeanVarianceOptimizer, 'RegimeSwitchingStrategy': RegimeSwitchingStrategy,
        'FractalMomentumStrategy': FractalMomentumStrategy, 'CopulaBlendStrategy': CopulaBlendStrategy,
        'WaveletDenoiser': WaveletDenoiser, 'GradientBoostBlender': GradientBoostBlender,
        'AttentionMechanism': AttentionMechanism, 'EnsembleVotingStrategy': EnsembleVotingStrategy,
        'OptimalTransportBlender': OptimalTransportBlender, 'StochasticDominance': StochasticDominance,
        'MaximumEntropyStrategy': MaximumEntropyStrategy, 'HiddenMarkovModel': HiddenMarkovModel,
        'QuantileRegressionStrategy': QuantileRegressionStrategy,
        'MutualInformationBlender': MutualInformationBlender,
        'GameTheoreticStrategy': GameTheoreticStrategy,
        'ReinforcementLearningInspired': ReinforcementLearningInspired,
        'SpectralClusteringStrategy': SpectralClusteringStrategy,
        'CausalInferenceStrategy': CausalInferenceStrategy,
        'BootstrapConfidenceStrategy': BootstrapConfidenceStrategy,
        'KernelDensityStrategy': KernelDensityStrategy,
        'SurvivalAnalysisStrategy': SurvivalAnalysisStrategy,
        'PrincipalComponentStrategy': PrincipalComponentStrategy,
        'FactorMomentumStrategy': FactorMomentumStrategy,
        'ElasticNetBlender': ElasticNetBlender,
        'RobustRegressionStrategy': RobustRegressionStrategy,
        'ConvexOptimizationStrategy': ConvexOptimizationStrategy,
        'MonteCarloStrategy': MonteCarloStrategy,
        'VariationalInferenceStrategy': VariationalInferenceStrategy,
        'NeuralNetworkInspired': NeuralNetworkInspired,
        'GraphNeuralInspired': GraphNeuralInspired,
        'ContrastiveLearningStrategy': ContrastiveLearningStrategy,
    }
    return {name: cls() for name, cls in strat_classes.items()}


# ══════════════════════════════════════════════════════════════════════════════
# CORE COMPUTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_portfolio_return(portfolio, next_prices):
    """Compute portfolio-weighted return between two periods."""
    if portfolio.empty or 'value' not in portfolio.columns or portfolio['value'].sum() == 0:
        return 0.0
    merged = portfolio.merge(next_prices[['symbol', 'price']], on='symbol', how='inner', suffixes=('_prev', '_next'))
    if merged.empty:
        return 0.0
    returns = (merged['price_next'] - merged['price_prev']) / merged['price_prev']
    return np.average(returns, weights=merged['value'])


def calculate_advanced_metrics(returns_with_dates):
    """
    Calculate comprehensive risk-adjusted performance metrics.
    Matches Pragyam 7.0.5 standards.
    """
    default_metrics = {
        'total_return': 0, 'annual_return': 0, 'volatility': 0,
        'sharpe': 0, 'sortino': 0, 'max_drawdown': 0, 'calmar': 0,
        'win_rate': 0, 'kelly_criterion': 0, 'omega_ratio': 1.0,
        'tail_ratio': 1.0, 'gain_to_pain': 0, 'profit_factor': 1.0
    }
    if len(returns_with_dates) < 2:
        return default_metrics, 52

    returns_df = pd.DataFrame(returns_with_dates).sort_values('date').set_index('date')
    time_deltas = returns_df.index.to_series().diff().dt.days
    avg_period_days = time_deltas.mean()
    periods_per_year = 365.25 / avg_period_days if pd.notna(avg_period_days) and avg_period_days > 0 else 52

    returns = returns_df['return']
    n_periods = len(returns)

    # Total Return (geometric)
    total_return = (1 + returns).prod() - 1

    # CAGR
    years = n_periods / periods_per_year
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = 0

    # Volatility (annualized standard deviation)
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)

    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe = annual_return / volatility if volatility > 0.001 else 0
    sharpe = np.clip(sharpe, -10, 10)

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) >= 2:
        downside_vol = downside_returns.std(ddof=1) * np.sqrt(periods_per_year)
        sortino = annual_return / downside_vol if downside_vol > 0.001 else 0
    else:
        sortino = 0
    sortino = np.clip(sortino, -20, 20)

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown_series = (cumulative / running_max) - 1
    max_drawdown = drawdown_series.min()

    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown < -0.001 else 0
    calmar = np.clip(calmar, -20, 20)

    # Win Rate
    win_rate = (returns > 0).mean()

    # Win/Loss Statistics
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = gains.mean() if len(gains) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    total_gains = gains.sum() if len(gains) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0

    # Kelly Criterion: f* = W - (1-W)/R where W=win_rate, R=avg_win/avg_loss
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0.0001 else 0
    kelly = (win_rate - ((1 - win_rate) / win_loss_ratio)) if win_loss_ratio > 0 else 0
    kelly = np.clip(kelly, -1, 1)

    # Omega Ratio
    omega_ratio = total_gains / total_losses if total_losses > 0.0001 else (total_gains * 10 if total_gains > 0 else 1.0)
    omega_ratio = np.clip(omega_ratio, 0, 50)

    # Profit Factor
    profit_factor = total_gains / total_losses if total_losses > 0.0001 else (10.0 if total_gains > 0 else 1.0)
    profit_factor = np.clip(profit_factor, 0, 50)

    # Tail Ratio
    upper_tail = np.percentile(returns, 95) if len(returns) >= 20 else returns.max()
    lower_tail = abs(np.percentile(returns, 5)) if len(returns) >= 20 else abs(returns.min())
    tail_ratio = upper_tail / lower_tail if lower_tail > 0.0001 else (10.0 if upper_tail > 0 else 1.0)
    tail_ratio = np.clip(tail_ratio, 0, 20)

    # Gain-to-Pain Ratio
    pain = abs(losses.sum()) if len(losses) > 0 else 0
    gain_to_pain = returns.sum() / pain if pain > 0.0001 else (returns.sum() * 10 if returns.sum() > 0 else 0)
    gain_to_pain = np.clip(gain_to_pain, -20, 20)

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'win_rate': win_rate,
        'kelly_criterion': kelly,
        'omega_ratio': omega_ratio,
        'tail_ratio': tail_ratio,
        'gain_to_pain': gain_to_pain,
        'profit_factor': profit_factor
    }
    return metrics, periods_per_year


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE (matches Pragyam 7.0.5 app.py _run_analysis)
# ══════════════════════════════════════════════════════════════════════════════

def run_pragyam_pipeline(
    investment_style: str,
    capital: float,
    num_positions: int = 30,
    callback=None
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Full Pragyam 7.0.5 pipeline: data fetch → regime → conviction-based curation.

    Architecture (2 phases):
      Phase 1: Data fetching + regime detection
      Phase 2: Conviction-based portfolio curation (ALL strategies)

    Args:
        investment_style: 'SIP Investment' or 'Swing Trading'
        capital: Amount in ₹
        num_positions: Number of positions to select (default: 30)
        callback: Optional fn(message, progress_pct) for status updates

    Returns:
        (portfolio_df, metadata_dict) or (None, error_dict)
    """
    # Reset metrics for fresh run
    reset_metrics()
    metrics = get_metrics()

    meta = {
        'investment_style': investment_style,
        'capital': capital,
        'num_positions': num_positions,
        'start_time': datetime.now().isoformat(),
        'phases': {},
        'regime': {},
    }

    def update(msg, pct=None):
        logger.info(msg)
        if callback:
            callback(msg, pct)

    try:
        # ═══════════════════════════════════════════════════════════
        # PHASE 1: DATA FETCHING + REGIME DETECTION
        # ═══════════════════════════════════════════════════════════
        update("━━ PHASE 1/2: DATA FETCHING ━━", 0.05)
        metrics.start_phase("data_fetching")

        lookback = 100
        end_date = datetime.now()
        total_days = int((lookback + MAX_INDICATOR_PERIOD) * 1.5) + 30
        fetch_start = end_date - timedelta(days=total_days)

        update(f"Downloading {len(SYMBOLS_UNIVERSE)} symbols from yfinance...", 0.08)

        try:
            historical_data = generate_historical_data(
                symbols_to_process=SYMBOLS_UNIVERSE,
                start_date=fetch_start,
                end_date=end_date,
            )
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            meta['error'] = str(e)
            metrics.end_phase("data_fetching", success=False, error_msg=str(e))
            return None, meta

        if not historical_data:
            meta['error'] = 'No historical data generated'
            metrics.end_phase("data_fetching", success=False, error_msg='Empty historical data')
            return None, meta

        metrics.end_phase("data_fetching", success=True, items=len(historical_data))
        metrics.days_count = len(historical_data)
        update(f"Loaded {len(historical_data)} trading days", 0.20)

        meta['phases']['data'] = {'days': len(historical_data)}

        # Get current data for portfolio curation
        current_date, current_df = historical_data[-1]
        training_window = historical_data[:-1]
        if len(training_window) > lookback:
            training_window = training_window[-lookback:]

        if len(historical_data) < 10:
            meta['error'] = f'Not enough training data ({len(historical_data)} days, need ≥10)'
            return None, meta

        # ─── Regime Detection ───
        update("━━ REGIME DETECTION ━━", 0.22)
        detector = MarketRegimeDetector()
        
        try:
            regime_result = detector.detect(historical_data, analysis_date=current_date)
            regime_dict = regime_result.to_dict()
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            regime_dict = {
                "regime": "UNKNOWN",
                "mix_name": "Chop/Consolidate Mix",
                "confidence": 0.30,
                "composite_score": 0.0,
                "explanation": f"Regime detection error: {e}",
                "color": "#6b7280",
                "icon": "❓",
                "description": "",
            }

        regime_name = regime_dict.get("regime", "UNKNOWN")
        mix_name = regime_dict.get("mix_name", "Chop/Consolidate Mix")
        confidence = regime_dict.get("confidence", 0.0)

        meta['regime'] = {
            'name': regime_name,
            'mix': mix_name,
            'confidence': confidence,
            'result_dict': regime_dict,
        }
        update(f"Regime: {regime_name} → {mix_name} ({confidence:.0%})", 0.25)

        # Store for later use
        meta['current_df'] = current_df
        meta['historical_data'] = historical_data

        # ═══════════════════════════════════════════════════════════
        # PHASE 2: CONVICTION-BASED CURATION (ALL STRATEGIES)
        # ═══════════════════════════════════════════════════════════
        update("━━ PHASE 2/2: CONVICTION-BASED CURATION ━━", 0.30)
        metrics.start_phase("conviction_curation")

        update("Loading strategies from registry...", 0.35)

        try:
            # Get ALL strategies (no filtering)
            strategies = get_all_strategies()
            # Exclude System_Curated if present
            strategies_to_run = {name: strategies[name] for name in strategies if name != "System_Curated"}

            if not strategies_to_run:
                meta['error'] = 'No strategies available'
                metrics.end_phase("conviction_curation", success=False, error_msg='Empty strategies')
                return None, meta

            update(f"Running ALL {len(strategies_to_run)} strategies...", 0.40)
            logger.info(f"Loaded {len(strategies_to_run)} strategies (excluded System_Curated)")

            # Aggregate holdings from ALL strategies
            update("Aggregating holdings across all strategies...", 0.50)
            aggregated_holdings = {}

            for idx, (name, strategy) in enumerate(strategies_to_run.items()):
                try:
                    port = strategy.generate_portfolio(current_df, capital)
                    if port.empty:
                        continue
                    
                    for _, row in port.iterrows():
                        symbol = row["symbol"]
                        price = row["price"]
                        if symbol not in aggregated_holdings:
                            aggregated_holdings[symbol] = {"price": price, "weight": 1.0}
                except Exception as e:
                    logger.debug(f"Strategy {name} produced no holdings: {e}")
                    continue

            if not aggregated_holdings:
                meta['error'] = 'No holdings generated from strategies'
                metrics.end_phase("conviction_curation", success=False, error_msg='No holdings generated')
                return None, meta

            update(f"Aggregated {len(aggregated_holdings)} unique candidate symbols", 0.65)

            # Conviction-based weighting with style-aware dispersion
            # SIP: +125% boost / -50% penalty | Swing: +225% boost / -75% penalty
            update("Computing conviction scores and applying style dispersion...", 0.70)

            min_pos_pct = 1.0 / 100  # 1%
            max_pos_pct = 10.0 / 100  # 10%

            try:
                portfolio_df = compute_conviction_based_weights(
                    aggregated_holdings,
                    current_df,
                    capital,
                    num_positions,
                    min_pos_pct,
                    max_pos_pct,
                    apply_dispersion=True,
                    investment_style=investment_style,  # Auto-selects dispersion based on style
                )
            except Exception as e:
                logger.error(f"Conviction-based weighting failed: {e}")
                meta['error'] = f'Portfolio weighting failed: {str(e)}'
                metrics.end_phase("conviction_curation", success=False, error_msg=str(e))
                return None, meta

            if portfolio_df.empty:
                meta['error'] = 'No portfolio generated after conviction weighting'
                metrics.end_phase("conviction_curation", success=False, error_msg='Empty portfolio')
                return None, meta

            # Calculate metadata
            total_value = portfolio_df['value'].sum()
            cash_remaining = capital - total_value
            avg_conviction = portfolio_df.get('conviction_score', pd.Series([50])).mean()
            top_conviction = portfolio_df.get('conviction_score', pd.Series([50])).max()

            meta['phases']['curation'] = {
                'positions': len(portfolio_df),
                'total_value': float(total_value),
                'cash_remaining': float(cash_remaining),
                'avg_conviction': float(avg_conviction),
                'top_conviction': float(top_conviction),
            }

            metrics.end_phase("conviction_curation", success=True)

            # Update metrics counters
            metrics.symbols_count = len(aggregated_holdings)
            metrics.strategies_count = len(strategies_to_run)
            metrics.portfolios_generated = len(portfolio_df)

            meta['analysis_date'] = current_date.strftime('%Y-%m-%d')
            meta['end_time'] = datetime.now().isoformat()

            update(f"✅ COMPLETE: {len(portfolio_df)} positions curated (avg conviction: {avg_conviction:.1f}/100)", 1.0)

            # Print execution summary
            metrics.print_summary()

            return portfolio_df, meta

        except Exception as e:
            logger.error(f"Conviction curation failed: {e}", exc_info=True)
            meta['error'] = str(e)
            metrics.end_phase("conviction_curation", success=False, error_msg=str(e))
            return None, meta

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        meta['error'] = str(e)
        return None, meta


__all__ = [
    "run_pragyam_pipeline",
    "compute_portfolio_return",
    "calculate_advanced_metrics",
    "get_all_strategies",
]
