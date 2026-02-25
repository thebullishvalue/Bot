"""
PRAGYAM Headless Engine
━━━━━━━━━━━━━━━━━━━━━━━
Runs the full Pragyam 4-phase pipeline without Streamlit dependencies.
Used by the Telegram bot to generate portfolios programmatically.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger("pragyam.engine")

# ─── Import Pragyam modules ───
from strategies import *
from backdata import (
    generate_historical_data,
    load_symbols_from_file,
    MAX_INDICATOR_PERIOD,
    SYMBOLS_UNIVERSE
)

try:
    from strategy_selection import (
        load_breadth_data,
        SIP_TRIGGER, SWING_BUY_TRIGGER, SWING_SELL_TRIGGER,
        BREADTH_SHEET_URL
    )
    STRATEGY_SELECTION_AVAILABLE = True
except ImportError:
    STRATEGY_SELECTION_AVAILABLE = False

try:
    from backtest_engine import (
        UnifiedBacktestEngine,
        DynamicPortfolioStylesGenerator,
        PerformanceMetrics
    )
    DYNAMIC_SELECTION_AVAILABLE = True
except ImportError:
    DYNAMIC_SELECTION_AVAILABLE = False

# ─── Trigger Config ───
TRIGGER_CONFIG = {
    'SIP Investment': {
        'buy_threshold': 0.42,
        'sell_threshold': 0.50,
        'sell_enabled': False,
        'description': 'Systematic accumulation on regime dips'
    },
    'Swing Trading': {
        'buy_threshold': 0.42,
        'sell_threshold': 0.50,
        'sell_enabled': True,
        'description': 'Tactical entry/exit on regime signals'
    },
}

PORTFOLIO_STYLES = {
    "Swing Trading": {
        "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility.",
        "mixes": {
            "Bull Market Mix": {
                "strategies": ['GameTheoreticStrategy', 'NebulaMomentumStorm', 'VolatilitySurfer', 'CelestialAlphaForge'],
            },
            "Bear Market Mix": {
                "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
            },
            "Chop/Consolidate Mix": {
                "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
            }
        }
    },
    "SIP Investment": {
        "description": "Systematic long-term wealth accumulation.",
        "mixes": {
            "Bull Market Mix": {
                "strategies": ['GameTheoreticStrategy', 'MomentumAccelerator', 'VolatilitySurfer', 'DivineMomentumOracle'],
            },
            "Bear Market Mix": {
                "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
            },
            "Chop/Consolidate Mix": {
                "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
            }
        }
    }
}


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


# ─── Market Regime Detection (from app.py) ───
class MarketRegimeDetectorV2:
    """
    Institutional-grade market regime detection (v2) with corrected scoring and
    classification logic.
    """
    
    def __init__(self):
        self.regime_thresholds = {
            'CRISIS': {'score': -1.0, 'confidence': 0.85},
            'BEAR': {'score': -0.5, 'confidence': 0.75},
            'WEAK_BEAR': {'score': -0.1, 'confidence': 0.65},
            'CHOP': {'score': 0.1, 'confidence': 0.60},
            'WEAK_BULL': {'score': 0.5, 'confidence': 0.65},
            'BULL': {'score': 1.0, 'confidence': 0.75},
            'STRONG_BULL': {'score': 1.5, 'confidence': 0.85},
        }
    
    def detect_regime(self, historical_data: list) -> Tuple[str, str, float, Dict]:
        if len(historical_data) < 10:
            return "INSUFFICIENT_DATA", "🐂 Bull Market Mix", 0.3, {}
        
        analysis_window = historical_data[-10:]
        latest_date, latest_df = analysis_window[-1]
        
        metrics = {
            'momentum': self._analyze_momentum_regime(analysis_window),
            'trend': self._analyze_trend_quality(analysis_window),
            'breadth': self._analyze_market_breadth(latest_df),
            'volatility': self._analyze_volatility_regime(analysis_window),
            'extremes': self._analyze_statistical_extremes(latest_df),
            'correlation': self._analyze_correlation_regime(latest_df),
            'velocity': self._analyze_velocity(analysis_window)
        }
        
        regime_score = self._calculate_composite_score(metrics)
        regime_name, confidence = self._classify_regime(regime_score, metrics)
        mix_name = self._map_regime_to_mix(regime_name)
        explanation = self._generate_explanation(regime_name, confidence, metrics, regime_score)
        
        return regime_name, mix_name, confidence, {
            'score': regime_score,
            'metrics': metrics,
            'explanation': explanation,
            'analysis_date': latest_date.strftime('%Y-%m-%d')
        }

    def _analyze_momentum_regime(self, window: list) -> Dict:
        rsi_values = [df['rsi latest'].mean() for _, df in window]
        osc_values = [df['osc latest'].mean() for _, df in window]
        
        current_rsi = rsi_values[-1]
        rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
        current_osc = osc_values[-1]
        osc_trend = np.polyfit(range(len(osc_values)), osc_values, 1)[0]
        
        if current_rsi > 65 and rsi_trend > 0.5:
            strength, score = 'STRONG_BULLISH', 2.0
        elif current_rsi > 55 and rsi_trend >= 0:
            strength, score = 'BULLISH', 1.0
        elif current_rsi < 35 and rsi_trend < -0.5:
            strength, score = 'STRONG_BEARISH', -2.0
        elif current_rsi < 45 and rsi_trend <= 0:
            strength, score = 'BEARISH', -1.0
        else:
            strength, score = 'NEUTRAL', 0.0
            
        return {'strength': strength, 'score': score, 'current_rsi': current_rsi, 'rsi_trend': rsi_trend, 'current_osc': current_osc, 'osc_trend': osc_trend}

    def _analyze_trend_quality(self, window: list) -> Dict:
        above_ma200_pct = [(df['price'] > df['ma200 latest']).mean() for _, df in window]
        ma_alignment = [(df['ma90 latest'] > df['ma200 latest']).mean() for _, df in window]
        
        current_above_200 = above_ma200_pct[-1]
        current_alignment = ma_alignment[-1]
        trend_consistency = np.polyfit(range(len(above_ma200_pct)), above_ma200_pct, 1)[0]
        
        if current_above_200 > 0.75 and current_alignment > 0.70 and trend_consistency >= 0:
            quality, score = 'STRONG_UPTREND', 2.0
        elif current_above_200 > 0.60 and current_alignment > 0.55:
            quality, score = 'UPTREND', 1.0
        elif current_above_200 < 0.30 and current_alignment < 0.30 and trend_consistency < 0:
            quality, score = 'STRONG_DOWNTREND', -2.0
        elif current_above_200 < 0.45 and current_alignment < 0.45:
            quality, score = 'DOWNTREND', -1.0
        else:
            quality, score = 'TRENDLESS', 0.0
            
        return {'quality': quality, 'score': score, 'above_200dma': current_above_200, 'ma_alignment': current_alignment, 'trend_consistency': trend_consistency}

    def _analyze_market_breadth(self, df: pd.DataFrame) -> Dict:
        rsi_bullish = (df['rsi latest'] > 50).mean()
        osc_positive = (df['osc latest'] > 0).mean()
        rsi_weak = (df['rsi latest'] < 40).mean()
        osc_oversold = (df['osc latest'] < -50).mean()
        divergence = abs(rsi_bullish - osc_positive)
        
        if rsi_bullish > 0.70 and osc_positive > 0.60 and divergence < 0.15:
            quality, score = 'STRONG_BROAD', 2.0
        elif rsi_bullish > 0.55 and osc_positive > 0.45:
            quality, score = 'HEALTHY', 1.0
        elif rsi_weak > 0.60 and osc_oversold > 0.50:
            quality, score = 'CAPITULATION', -2.0
        elif rsi_weak > 0.45 and osc_oversold > 0.35:
            quality, score = 'WEAK', -1.0
        elif divergence > 0.25:
            quality, score = 'DIVERGENT', -0.5
        else:
            quality, score = 'MIXED', 0.0
            
        return {'quality': quality, 'score': score, 'rsi_bullish_pct': rsi_bullish, 'osc_positive_pct': osc_positive, 'divergence': divergence}

    def _analyze_volatility_regime(self, window: list) -> Dict:
        bb_widths = [((4 * df['dev20 latest']) / (df['ma20 latest'] + 1e-6)).mean() for _, df in window]
        current_bbw = bb_widths[-1]
        vol_trend = np.polyfit(range(len(bb_widths)), bb_widths, 1)[0]
        
        if current_bbw < 0.08 and vol_trend < 0:
            regime, score = 'SQUEEZE', 0.5 
        elif current_bbw > 0.15 and vol_trend > 0:
            regime, score = 'PANIC', -1.0 
        elif current_bbw > 0.12:
            regime, score = 'ELEVATED', -0.5
        else:
            regime, score = 'NORMAL', 0.0
            
        return {'regime': regime, 'score': score, 'current_bbw': current_bbw, 'vol_trend': vol_trend}

    def _analyze_statistical_extremes(self, df: pd.DataFrame) -> Dict:
        extreme_oversold = (df['zscore latest'] < -2.0).mean()
        extreme_overbought = (df['zscore latest'] > 2.0).mean()
        
        if extreme_oversold > 0.40:
            extreme_type, score = 'DEEPLY_OVERSOLD', 1.5 
        elif extreme_overbought > 0.40:
            extreme_type, score = 'DEEPLY_OVERBOUGHT', -1.5
        elif extreme_oversold > 0.20:
            extreme_type, score = 'OVERSOLD', 0.75
        elif extreme_overbought > 0.20:
            extreme_type, score = 'OVERBOUGHT', -0.75
        else:
            extreme_type, score = 'NORMAL', 0.0
            
        return {'type': extreme_type, 'score': score, 'zscore_extreme_oversold_pct': extreme_oversold, 'zscore_extreme_overbought_pct': extreme_overbought}

    def _analyze_correlation_regime(self, df: pd.DataFrame) -> Dict:
        """
        Analyze cross-sectional correlation structure.
        
        High correlation (herding) often precedes market stress.
        Low correlation indicates stock-picking environment.
        
        Mathematical approach: Compute pairwise correlation proxy via 
        indicator agreement across the cross-section.
        """
        # Cross-sectional correlation proxy via indicator agreement
        # When indicators agree across stocks, correlation is high
        rsi_median = df['rsi latest'].median()
        osc_median = df['osc latest'].median()
        
        # Fraction of stocks on same side of median (herding measure)
        rsi_above = (df['rsi latest'] > rsi_median).mean()
        rsi_agreement = max(rsi_above, 1 - rsi_above)  # Closer to 1 = more agreement
        
        osc_above = (df['osc latest'] > osc_median).mean()
        osc_agreement = max(osc_above, 1 - osc_above)
        
        # Cross-indicator agreement (both oversold or both overbought)
        both_oversold = ((df['rsi latest'] < 40) & (df['osc latest'] < -30)).mean()
        both_overbought = ((df['rsi latest'] > 60) & (df['osc latest'] > 30)).mean()
        indicator_agreement = both_oversold + both_overbought
        
        # Dispersion as inverse correlation proxy
        rsi_dispersion = df['rsi latest'].std() / 50  # Normalized
        osc_dispersion = df['osc latest'].std() / 100
        avg_dispersion = (rsi_dispersion + osc_dispersion) / 2
        
        # Combined correlation score (0 = dispersed, 1 = correlated)
        correlation_score = (rsi_agreement + osc_agreement) / 2 * (1 - avg_dispersion) + indicator_agreement * 0.3
        correlation_score = np.clip(correlation_score, 0, 1)
        
        if correlation_score > 0.7:
            regime, score = 'HIGH_CORRELATION', -0.5  # High corr often precedes stress
        elif correlation_score < 0.4:
            regime, score = 'LOW_CORRELATION', 0.5  # Good for stock picking
        else:
            regime, score = 'NORMAL', 0.0
            
        return {
            'regime': regime, 
            'score': score, 
            'correlation_score': correlation_score,
            'dispersion': avg_dispersion,
            'indicator_agreement': indicator_agreement
        }

    def _analyze_velocity(self, window: list) -> Dict:
        """
        Analyze momentum velocity and acceleration.
        
        Velocity: First derivative of RSI (rate of change)
        Acceleration: Second derivative (rate of change of velocity)
        
        Positive acceleration with positive velocity = strengthening momentum
        Negative acceleration with positive velocity = momentum fading
        """
        if len(window) < 5: 
            return {'acceleration': 'UNKNOWN', 'score': 0.0, 'avg_velocity': 0.0, 'acceleration_value': 0.0}
        
        recent_rsis = np.array([w[1]['rsi latest'].mean() for w in window[-5:]])
        
        # Velocity: First differences (first derivative)
        velocity = np.diff(recent_rsis)  # 4 values
        avg_velocity = np.mean(velocity)
        current_velocity = velocity[-1]
        
        # Acceleration: Second differences (second derivative)
        acceleration_values = np.diff(velocity)  # 3 values
        avg_acceleration = np.mean(acceleration_values)
        current_acceleration = acceleration_values[-1]
        
        # Classification based on velocity and acceleration
        if avg_velocity > 1.5 and current_acceleration > 0:
            velocity_regime, score = 'ACCELERATING_UP', 1.5
        elif avg_velocity > 1.0 and current_acceleration >= 0:
            velocity_regime, score = 'RISING_FAST', 1.0
        elif avg_velocity > 0.5:
            velocity_regime, score = 'RISING', 0.5
        elif avg_velocity < -1.5 and current_acceleration < 0:
            velocity_regime, score = 'ACCELERATING_DOWN', -1.5
        elif avg_velocity < -1.0 and current_acceleration <= 0:
            velocity_regime, score = 'FALLING_FAST', -1.0
        elif avg_velocity < -0.5:
            velocity_regime, score = 'FALLING', -0.5
        elif abs(avg_velocity) < 0.5 and abs(current_acceleration) > 0.5:
            # Momentum building from stable base
            if current_acceleration > 0:
                velocity_regime, score = 'COILING_UP', 0.3
            else:
                velocity_regime, score = 'COILING_DOWN', -0.3
        else:
            velocity_regime, score = 'STABLE', 0.0
            
        return {
            'acceleration': velocity_regime, 
            'score': score, 
            'avg_velocity': avg_velocity,
            'current_velocity': current_velocity,
            'acceleration_value': current_acceleration
        }

    def _calculate_composite_score(self, metrics: Dict) -> float:
        weights = { 'momentum': 0.30, 'trend': 0.25, 'breadth': 0.15, 'volatility': 0.05, 'extremes': 0.10, 'correlation': 0.0, 'velocity': 0.15 }
        return sum(metrics[factor]['score'] * weight for factor, weight in weights.items())
    
    def _classify_regime(self, score: float, metrics: Dict) -> Tuple[str, float]:
        if metrics['volatility']['regime'] == 'PANIC' and score < -0.5 and metrics['breadth']['quality'] == 'CAPITULATION':
            return 'CRISIS', 0.90
            
        sorted_thresholds = sorted(self.regime_thresholds.items(), key=lambda item: item[1]['score'])
        
        for regime, threshold in reversed(sorted_thresholds):
            if score >= threshold['score']:
                confidence = threshold['confidence'] * 0.75 if metrics['breadth']['quality'] == 'DIVERGENT' else threshold['confidence']
                return regime, confidence

        return 'CRISIS', 0.85
    
    def _map_regime_to_mix(self, regime: str) -> str:
        mapping = {
            'STRONG_BULL': 'Bull Market Mix', 'BULL': 'Bull Market Mix',
            'WEAK_BULL': 'Chop/Consolidate Mix', 'CHOP': 'Chop/Consolidate Mix',
            'WEAK_BEAR': 'Chop/Consolidate Mix', 'BEAR': 'Bear Market Mix',
            'CRISIS': 'Bear Market Mix'
        }
        return mapping.get(regime, 'Chop/Consolidate Mix')
    
    def _generate_explanation(self, regime: str, confidence: float, metrics: Dict, score: float) -> str:
        lines = [f"**Detected Regime:** {regime} (Score: {score:.2f}, Confidence: {confidence:.0%})", ""]
        rationales = {
            'STRONG_BULL': "Strong upward momentum with broad participation. Favor momentum strategies.",
            'BULL': "Positive trend with healthy breadth. Conditions support growth strategies.",
            'WEAK_BULL': "Uptrend showing signs of fatigue or divergence. Rotate to defensive positions.",
            'CHOP': "No clear directional bias. Favors mean reversion and relative value strategies.",
            'WEAK_BEAR': "Downtrend developing. Begin defensive positioning.",
            'BEAR': "Established downtrend with weak breadth. Favor defensive strategies.",
            'CRISIS': "Severe market stress. Focus on capital preservation and oversold opportunities."
        }
        lines.append(f"**Rationale:** {rationales.get(regime, 'Market conditions unclear.')}")
        if metrics['breadth']['quality'] == 'DIVERGENT':
            lines.append("⚠️ **Warning:** Breadth divergence detected - narrow leadership may not be sustainable.")
        lines.append("\n**Key Factors:**")
        lines.append(f"• **Momentum:** {metrics['momentum']['strength']} (RSI: {metrics['momentum']['current_rsi']:.1f})")
        lines.append(f"• **Trend:** {metrics['trend']['quality']} ({metrics['trend']['above_200dma']:.0%} > 200DMA)")
        lines.append(f"• **Breadth:** {metrics['breadth']['quality']} ({metrics['breadth']['rsi_bullish_pct']:.0%} bullish)")
        lines.append(f"• **Volatility:** {metrics['volatility']['regime']} (BBW: {metrics['volatility']['current_bbw']:.3f})")
        if metrics['extremes']['type'] != 'NORMAL':
            lines.append(f"• **Extremes:** {metrics['extremes']['type']} detected")
        return "\n".join(lines)


# ─── Core computation functions (headless versions) ───

def compute_portfolio_return(portfolio, next_prices):
    if portfolio.empty or 'value' not in portfolio.columns or portfolio['value'].sum() == 0:
        return 0.0
    merged = portfolio.merge(next_prices[['symbol', 'price']], on='symbol', how='inner', suffixes=('_prev', '_next'))
    if merged.empty: return 0.0
    returns = (merged['price_next'] - merged['price_prev']) / merged['price_prev']
    return np.average(returns, weights=merged['value'])


def calculate_advanced_metrics(returns_with_dates):
    """
    Calculate comprehensive risk-adjusted performance metrics.
    Matches Pragyam app.py calculate_advanced_metrics exactly.
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


def calculate_strategy_weights(performance):
    names = list(performance['strategy'].keys())
    if not names: return {}
    sharpes = []
    for n in names:
        d = performance['strategy'][n]
        if isinstance(d, dict) and 'metrics' in d:
            s = d['metrics'].get('sharpe', 0)
        else:
            s = d.get('sharpe', 0)
        if not isinstance(s, (int, float)) or not np.isfinite(s): s = 0
        sharpes.append(s + 2)
    sharpes = np.array(sharpes)
    stable = sharpes - np.max(sharpes)
    exp_s = np.exp(stable)
    total = np.sum(exp_s)
    if total == 0 or not np.isfinite(total):
        return {n: 1.0 / len(names) for n in names}
    w = exp_s / total
    return {n: w[i] for i, n in enumerate(names)}


def _calc_perf_on_window(window_data, strategies, capital):
    perf = {n: {'returns': []} for n in strategies}
    subset_perf = {n: {} for n in strategies}
    for i in range(len(window_data) - 1):
        date, df = window_data[i]
        _, ndf = window_data[i + 1]
        for name, strat in strategies.items():
            try:
                port = strat.generate_portfolio(df, capital)
                if port.empty: continue
                perf[name]['returns'].append({'return': compute_portfolio_return(port, ndf), 'date': _})
                n_rows, ts = len(port), 10
                nt = n_rows // ts
                for j in range(nt):
                    tn = f'tier_{j+1}'
                    if tn not in subset_perf[name]: subset_perf[name][tn] = []
                    sub = port.iloc[j*ts:(j+1)*ts]
                    if not sub.empty:
                        subset_perf[name][tn].append({'return': compute_portfolio_return(sub, ndf), 'date': _})
            except Exception as e:
                logger.error(f"Window calc error ({name}): {e}")
    fp = {}
    for name, data in perf.items():
        m, _ = calculate_advanced_metrics(data['returns'])
        fp[name] = {'metrics': m, 'sharpe': m['sharpe']}
    fs = {}
    for name, data in subset_perf.items():
        fs[name] = {}
        for sub, sp in data.items():
            if sp:
                fs[name][sub] = calculate_advanced_metrics(sp)[0]['sharpe']
    return {'strategy': fp, 'subset': fs}


def evaluate_historical_performance_headless(strategies, historical_data, callback=None):
    """Walk-forward evaluation without Streamlit."""
    MIN_TRAIN = 2
    CAPITAL = 2500000.0
    if len(historical_data) < MIN_TRAIN + 1:
        logger.error("Not enough historical data for backtest")
        return {}

    all_names = list(strategies.keys()) + ['System_Curated']
    oos = {n: {'returns': []} for n in all_names}
    total_steps = len(historical_data) - MIN_TRAIN - 1
    if total_steps <= 0: return {}

    for i in range(MIN_TRAIN, len(historical_data) - 1):
        train = historical_data[:i]
        test_date, test_df = historical_data[i]
        next_date, next_df = historical_data[i + 1]

        step = i - MIN_TRAIN + 1
        if callback:
            callback(f"Walk-forward step {step}/{total_steps}", step / total_steps)

        in_sample = _calc_perf_on_window(train, strategies, CAPITAL)
        try:
            curated, sw, subw = curate_final_portfolio(strategies, in_sample, test_df, CAPITAL, 30, 1.0, 10.0)
            if curated.empty:
                oos['System_Curated']['returns'].append({'return': 0, 'date': next_date})
            else:
                oos['System_Curated']['returns'].append({
                    'return': compute_portfolio_return(curated, next_df), 'date': next_date
                })
        except Exception as e:
            logger.error(f"OOS curation error: {e}")
            oos['System_Curated']['returns'].append({'return': 0, 'date': next_date})

        for name, strat in strategies.items():
            try:
                port = strat.generate_portfolio(test_df, CAPITAL)
                oos[name]['returns'].append({
                    'return': compute_portfolio_return(port, next_df), 'date': next_date
                })
            except:
                oos[name]['returns'].append({'return': 0, 'date': next_date})

    final = {}
    for name, data in oos.items():
        m, _ = calculate_advanced_metrics(data['returns'])
        final[name] = {'returns': data['returns'], 'metrics': m}

    full_subset = _calc_perf_on_window(historical_data, strategies, CAPITAL)['subset']
    return {'strategy': final, 'subset': full_subset}


def curate_final_portfolio(strategies, performance, current_df, capital, num_positions, min_pct, max_pct):
    sw = calculate_strategy_weights(performance)
    subset_w = {}
    for name in strategies:
        sp = performance.get('subset', {}).get(name, {})
        tiers = sorted(sp.keys())
        if not tiers:
            subset_w[name] = {}
            continue
        ts = np.array([sp.get(t, 1.0 - int(t.split('_')[1]) * 0.05) + 2 for t in tiers])
        stable = ts - np.max(ts)
        exp_t = np.exp(stable)
        total = np.sum(exp_t)
        if total > 0 and np.isfinite(total):
            subset_w[name] = {t: exp_t[i] / total for i, t in enumerate(tiers)}
        else:
            eq = 1.0 / len(tiers)
            subset_w[name] = {t: eq for t in tiers}

    agg = {}
    for name, strat in strategies.items():
        port = strat.generate_portfolio(current_df, capital)
        if port.empty: continue
        n, ts = len(port), 10
        nt = n // ts
        if nt == 0: continue
        for j in range(nt):
            tn = f'tier_{j+1}'
            if tn not in subset_w.get(name, {}): continue
            sub = port.iloc[j*ts:(j+1)*ts]
            tw = subset_w[name][tn]
            for _, row in sub.iterrows():
                sym, price, wpct = row['symbol'], row['price'], row['weightage_pct']
                fw = (wpct / 100) * tw * sw.get(name, 0)
                if sym in agg: agg[sym]['weight'] += fw
                else: agg[sym] = {'price': price, 'weight': fw}

    if not agg: return pd.DataFrame(), {}, {}
    fp = pd.DataFrame([{'symbol': s, **d} for s, d in agg.items()]).sort_values('weight', ascending=False).head(num_positions)
    tw = fp['weight'].sum()
    fp['weightage_pct'] = fp['weight'] * 100 / tw
    fp['weightage_pct'] = fp['weightage_pct'].clip(lower=min_pct, upper=max_pct)
    fp['weightage_pct'] = (fp['weightage_pct'] / fp['weightage_pct'].sum()) * 100
    fp['units'] = np.floor((capital * fp['weightage_pct'] / 100) / fp['price'])
    fp['value'] = fp['units'] * fp['price']
    return fp.sort_values('weightage_pct', ascending=False).reset_index(drop=True), sw, subset_w


def _compute_backtest_metrics(daily_values, periods_per_year=252.0):
    """
    Compute performance metrics from daily portfolio values.
    Returns realistic, unbounded metrics for proper comparison.
    Matches Pragyam app.py _compute_backtest_metrics exactly.
    """
    result = {
        'total_return': 0.0,
        'ann_return': 0.0,
        'volatility': 0.0,
        'sharpe': 0.0,
        'sortino': 0.0,
        'calmar': 0.0,
        'max_dd': 0.0,
        'win_rate': 0.0
    }

    if len(daily_values) < 5:
        return result

    values = np.array(daily_values, dtype=np.float64)

    # Validate data
    if np.any(values <= 0) or np.any(~np.isfinite(values)):
        return result

    initial = values[0]
    final = values[-1]
    n_days = len(values)

    # Total Return
    total_return = (final - initial) / initial
    result['total_return'] = total_return

    # Daily Returns
    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    if len(daily_returns) < 3:
        return result

    # Annualized Return (CAGR)
    years = n_days / periods_per_year
    if years > 0 and final > 0 and initial > 0:
        ann_return = (final / initial) ** (1.0 / years) - 1.0
    else:
        ann_return = 0.0
    result['ann_return'] = ann_return

    # Volatility (annualized)
    daily_vol = np.std(daily_returns, ddof=1)
    volatility = daily_vol * np.sqrt(periods_per_year)
    result['volatility'] = volatility

    # Sharpe Ratio
    if volatility > 0.001:
        sharpe = ann_return / volatility
    else:
        sharpe = 0.0
    sharpe = np.clip(sharpe, -10, 10)
    result['sharpe'] = sharpe

    # Sortino Ratio (downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) >= 2:
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(periods_per_year)
        if downside_vol > 0.001:
            sortino = ann_return / downside_vol
        else:
            sortino = 0
    else:
        sortino = 0
    sortino = np.clip(sortino, -20, 20)
    result['sortino'] = sortino

    # Maximum Drawdown
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_dd = np.min(drawdowns)
    result['max_dd'] = max_dd

    # Calmar Ratio (annualized return / max drawdown)
    if max_dd < -0.001:  # At least 0.1% drawdown
        calmar = ann_return / abs(max_dd)
    else:
        calmar = 0
    calmar = np.clip(calmar, -20, 20)
    result['calmar'] = calmar

    # Win Rate
    win_rate = np.mean(daily_returns > 0)
    result['win_rate'] = win_rate

    return result


def run_dynamic_strategy_selection_headless(historical_data, all_strategies, selected_style,
                                            trigger_df=None, trigger_config=None, callback=None):
    """
    Headless version of _run_dynamic_strategy_selection.
    Matches Pragyam app.py trigger-based backtest logic exactly.
    """
    is_sip = "SIP" in selected_style
    metric_key = 'calmar' if is_sip else 'sortino'

    if trigger_config is None:
        trigger_config = TRIGGER_CONFIG.get(selected_style, TRIGGER_CONFIG.get('SIP Investment', {}))

    buy_threshold = trigger_config.get('buy_threshold', 0.42 if is_sip else 0.52)
    sell_threshold = trigger_config.get('sell_threshold', 1.5 if is_sip else 1.2)
    sell_enabled = trigger_config.get('sell_enabled', not is_sip)

    if not historical_data or len(historical_data) < 10:
        return None, {}

    date_to_df = {}
    simulation_dates = []
    for date_obj, df in historical_data:
        sd = date_obj.date() if hasattr(date_obj, 'date') else date_obj
        simulation_dates.append(sd)
        date_to_df[sd] = df

    all_symbols = set()
    for _, df in historical_data:
        all_symbols.update(df['symbol'].tolist())
    all_symbols = sorted(all_symbols)
    n_days = len(historical_data)
    capital = 10_000_000

    # Build trigger masks
    buy_dates_mask = [False] * n_days
    sell_dates_mask = [False] * n_days
    if trigger_df is not None and not trigger_df.empty and 'REL_BREADTH' in trigger_df.columns:
        if hasattr(trigger_df.index, 'date'):
            trigger_date_map = {idx.date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}
        else:
            trigger_date_map = {pd.to_datetime(idx).date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}
        for i, sim_date in enumerate(simulation_dates):
            if sim_date in trigger_date_map:
                rel_breadth = trigger_date_map[sim_date]
                if rel_breadth < buy_threshold:
                    buy_dates_mask[i] = True
                if sell_enabled and rel_breadth > sell_threshold:
                    sell_dates_mask[i] = True
    else:
        buy_dates_mask[0] = True

    results = {}
    valid_strategies = []
    total = len(all_strategies)

    for idx, (name, strategy) in enumerate(all_strategies.items()):
        if callback:
            callback(f"Backtesting: {name} ({idx+1}/{total})", 0.25 + (idx/total) * 0.35)

        try:
            daily_values = []
            portfolio_units = {}
            buy_signal_active = False
            trade_log = []

            if is_sip:
                # ── SIP MODE: Accumulate on each buy trigger, track TWR ──
                nav_index = 1.0
                prev_portfolio_value = 0.0
                has_position = False
                sip_amount = capital

                for j, sim_date in enumerate(simulation_dates):
                    df = date_to_df[sim_date]
                    prices_today = df.set_index('symbol')['price']

                    # Step 1: Compute current value of EXISTING holdings
                    current_value = 0.0
                    if portfolio_units:
                        current_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )

                    # Step 2: Update NAV based on market movement BEFORE any new investment
                    if has_position and prev_portfolio_value > 0:
                        day_return = (current_value - prev_portfolio_value) / prev_portfolio_value
                        nav_index *= (1 + day_return)

                    # Step 3: Check buy/sell triggers
                    is_buy_day = buy_dates_mask[j]
                    actual_buy_trigger = is_buy_day and not buy_signal_active

                    if is_buy_day:
                        buy_signal_active = True
                    else:
                        buy_signal_active = False

                    # Sell (SIP rarely sells, but support it)
                    if sell_dates_mask[j] and portfolio_units and sell_enabled:
                        trade_log.append({'Event': 'SELL', 'Date': sim_date})
                        portfolio_units = {}
                        has_position = False
                        current_value = 0.0

                    # Step 4: Execute SIP buy (does NOT affect nav_index — TWR principle)
                    if actual_buy_trigger:
                        trade_log.append({'Event': 'BUY', 'Date': sim_date})
                        buy_portfolio = strategy.generate_portfolio(df.copy(), sip_amount)

                        if buy_portfolio is not None and not buy_portfolio.empty and 'value' in buy_portfolio.columns:
                            for _, row in buy_portfolio.iterrows():
                                sym = row['symbol']
                                u = row.get('units', 0)
                                if u > 0:
                                    portfolio_units[sym] = portfolio_units.get(sym, 0) + u
                            has_position = True

                            # Recalculate value after addition for next day's return base
                            current_value = sum(
                                units * prices_today.get(sym, 0)
                                for sym, units in portfolio_units.items()
                            )

                    prev_portfolio_value = current_value
                    daily_values.append(nav_index)

            else:
                # ── SWING MODE: Single position, hold until sell trigger ──
                current_capital = capital

                for j, sim_date in enumerate(simulation_dates):
                    df = date_to_df[sim_date]

                    is_buy_day = buy_dates_mask[j]
                    actual_buy_trigger = is_buy_day and not buy_signal_active

                    if is_buy_day:
                        buy_signal_active = True
                    else:
                        buy_signal_active = False

                    # Check sell trigger
                    if sell_dates_mask[j] and portfolio_units:
                        trade_log.append({'Event': 'SELL', 'Date': sim_date})
                        prices_today = df.set_index('symbol')['price']
                        sell_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )
                        current_capital += sell_value
                        portfolio_units = {}
                        buy_signal_active = False

                    # Execute buy (only if no position)
                    if actual_buy_trigger and not portfolio_units and current_capital > 1000:
                        trade_log.append({'Event': 'BUY', 'Date': sim_date})
                        buy_portfolio = strategy.generate_portfolio(df.copy(), current_capital)

                        if buy_portfolio is not None and not buy_portfolio.empty and 'units' in buy_portfolio.columns:
                            portfolio_units = pd.Series(
                                buy_portfolio['units'].values,
                                index=buy_portfolio['symbol']
                            ).to_dict()
                            current_capital -= buy_portfolio['value'].sum()

                    # Calculate current value
                    portfolio_value = 0
                    if portfolio_units:
                        prices_today = df.set_index('symbol')['price']
                        portfolio_value = sum(
                            units * prices_today.get(sym, 0)
                            for sym, units in portfolio_units.items()
                        )

                    daily_values.append(portfolio_value + current_capital)

            # ── COMPUTE METRICS ──
            if len(daily_values) < 10 or daily_values[0] <= 0:
                logger.debug(f"  {name}: Invalid daily values - SKIP")
                results[name] = {'status': 'skip', 'reason': 'Invalid values'}
                continue

            metrics = _compute_backtest_metrics(daily_values)

            score = metrics[metric_key]

            # Add trade info
            metrics['buy_events'] = len([t for t in trade_log if t['Event'] == 'BUY'])
            metrics['sell_events'] = len([t for t in trade_log if t['Event'] == 'SELL'])
            metrics['trade_events'] = len(trade_log)

            # Validate score
            if not np.isfinite(score):
                logger.debug(f"  {name}: Invalid {metric_key} ({score}) - SKIP")
                results[name] = {'status': 'skip', 'reason': f'Invalid {metric_key}'}
                continue

            # Store results
            results[name] = {
                'status': 'ok',
                'metrics': metrics,
                'score': score,
                'positions': len(portfolio_units) if portfolio_units else 0,
                'trade_log': trade_log
            }
            valid_strategies.append((name, score, metrics))

        except Exception as e:
            logger.error(f"Strategy backtest error ({name}): {e}")
            results[name] = {'status': 'error', 'reason': str(e)}
            continue

    # Select top 4
    if len(valid_strategies) < 4:
        logger.warning(f"Only {len(valid_strategies)} valid strategies (need 4) - using static selection")
        return None, results

    valid_strategies.sort(key=lambda x: x[1], reverse=True)
    top4 = [name for name, _, _ in valid_strategies[:4]]
    logger.info(f"Dynamic selection: Top 4 by {metric_key}: {top4}")
    return top4, results


# ─── Main Pipeline ───

def run_pragyam_pipeline(investment_style: str, capital: float, callback=None) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Full Pragyam pipeline: data fetch → regime → strategy selection → walk-forward → curation.
    
    Args:
        investment_style: 'SIP Investment' or 'Swing Trading'
        capital: Amount in ₹
        callback: Optional fn(message, progress_pct) for status updates
    
    Returns:
        (portfolio_df, metadata_dict) or (None, error_dict)
    """
    meta = {
        'investment_style': investment_style,
        'capital': capital,
        'start_time': datetime.now().isoformat(),
        'phases': {},
    }

    def update(msg, pct=None):
        logger.info(msg)
        if callback:
            callback(msg, pct)

    update("━━ PHASE 1/4: DATA FETCHING ━━", 0.05)
    
    lookback = 100
    end_date = datetime.now()
    total_days = int((lookback + MAX_INDICATOR_PERIOD) * 12)
    fetch_start = end_date - timedelta(days=total_days)

    update(f"Downloading {len(SYMBOLS_UNIVERSE)} symbols...", 0.08)
    
    try:
        historical_data = generate_historical_data(SYMBOLS_UNIVERSE, fetch_start, end_date)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        meta['error'] = str(e)
        return None, meta

    if not historical_data:
        meta['error'] = 'No historical data generated'
        return None, meta

    update(f"Loaded {len(historical_data)} trading days", 0.20)
    meta['phases']['data'] = {'days': len(historical_data)}

    current_date, current_df = historical_data[-1]
    training_data = historical_data[:-1]
    if len(training_data) > lookback:
        training_window = training_data[-lookback:]
    else:
        training_window = training_data
    training_window_with_current = training_window + [(current_date, current_df)]

    if len(training_window_with_current) < 10:
        meta['error'] = f'Not enough training data ({len(training_window_with_current)} days)'
        return None, meta

    # ─── Phase 1.5: Regime Detection ───
    update("━━ REGIME DETECTION ━━", 0.22)
    detector = MarketRegimeDetectorV2()
    regime_name, mix_name, confidence, details = detector.detect_regime(historical_data)
    meta['regime'] = {'name': regime_name, 'mix': mix_name, 'confidence': confidence}
    update(f"Regime: {regime_name} → {mix_name} ({confidence:.0%})", 0.25)

    # ─── Phase 2: Dynamic Strategy Selection ───
    update("━━ PHASE 2/4: STRATEGY SELECTION ━━", 0.25)
    strategies = get_all_strategies()

    trigger_df = None
    if STRATEGY_SELECTION_AVAILABLE:
        try:
            breadth_df = load_breadth_data(lookback_rows=600)
            if not breadth_df.empty:
                trigger_df = breadth_df.copy().set_index('DATE')
                update(f"Loaded {len(trigger_df)} trigger entries", 0.28)
        except Exception as e:
            logger.warning(f"Breadth data fetch failed: {e}")

    trigger_config = TRIGGER_CONFIG.get(investment_style, TRIGGER_CONFIG['SIP Investment'])

    dynamic_strats, strat_metrics = run_dynamic_strategy_selection_headless(
        training_window_with_current, strategies, investment_style,
        trigger_df=trigger_df, trigger_config=trigger_config, callback=callback
    )

    if dynamic_strats and len(dynamic_strats) >= 4:
        style_strategies = dynamic_strats
        selection_mode = "DYNAMIC"
    else:
        style_strategies = PORTFOLIO_STYLES[investment_style]["mixes"][mix_name]['strategies']
        selection_mode = "STATIC"

    strategies_to_run = {n: strategies[n] for n in style_strategies if n in strategies}
    if not strategies_to_run:
        meta['error'] = 'No valid strategies available'
        return None, meta

    meta['phases']['selection'] = {'mode': selection_mode, 'strategies': list(strategies_to_run.keys())}
    update(f"Selected {len(strategies_to_run)} strategies ({selection_mode})", 0.60)

    # ─── Phase 3: Walk-Forward Evaluation ───
    update("━━ PHASE 3/4: WALK-FORWARD EVALUATION ━━", 0.65)
    PHASE3_LOOKBACK = 50
    phase3_data = training_window_with_current[-PHASE3_LOOKBACK:] if len(training_window_with_current) > PHASE3_LOOKBACK else training_window_with_current

    performance = evaluate_historical_performance_headless(strategies_to_run, phase3_data, callback=callback)
    if not performance:
        meta['error'] = 'Walk-forward evaluation produced no results'
        return None, meta

    # ─── Phase 4: Portfolio Curation ───
    update("━━ PHASE 4/4: PORTFOLIO CURATION ━━", 0.90)
    num_positions = 30
    min_pct, max_pct = 1.0, 10.0

    portfolio_df, sw, subw = curate_final_portfolio(
        strategies_to_run, performance, current_df, capital, num_positions, min_pct, max_pct
    )

    if portfolio_df.empty:
        meta['error'] = 'Portfolio curation returned empty'
        return None, meta

    total_value = portfolio_df['value'].sum()
    cash_remaining = capital - total_value

    meta['phases']['curation'] = {
        'positions': len(portfolio_df),
        'total_value': float(total_value),
        'cash_remaining': float(cash_remaining),
    }
    meta['analysis_date'] = current_date.strftime('%Y-%m-%d')
    meta['end_time'] = datetime.now().isoformat()

    update(f"✅ COMPLETE: {len(portfolio_df)} positions curated", 1.0)
    return portfolio_df, meta
