# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

"""
SurfMultiModel v1.26: Quality-Mirrored Exits (Entry/Exit Symmetry)

What's New in v1.26:
- QUALITY-MIRRORED EXITS: Exit when OPPOSITE direction quality score is high (70% of entry threshold)
- PERFECT SYMMETRY: Long exit at peak when short_score high, Short exit at trough when long_score high
- EXTREMA ALIGNMENT: Enter at immediate extrema (≤1), Exit in wider window (≤3)
- MULTI-CRITERIA EXITS: Opposite quality score OR omega flip OR (accel + complexity)
- TARGET: 2.5-3.0x exit/entry ratio with quality-based exits
- AGGRESSIVE ROI: 2.0% immediate, 0.5% at 90min (vs 3.5%/1.0% in v1.23)
- MULTI-FACTOR SIZING: Omega × consistency × trend (0.3x to 2.5x)

Core Innovation: Omega as Mandelbrot Approximator
- Fat tails: Captured in sums (not variance)
- Asymmetry: Separate gains/losses (true asymmetry)
- Non-Gaussian: Distribution-free (works for any distribution)
- Scale invariance: Multi-scale Omega reveals self-similarity

Why This Works:
- Omega ratio IS a Mandelbrot approximator (captures fat tails, asymmetry, non-Gaussian)
- Multi-scale reveals fractal properties (scale consistency)
- Quality prediction (not just direction) → Higher win rate
- Simpler exit logic prevents under-exiting while avoiding overproliferation

Author: Strategy Evolution Framework
Date: October 7, 2025
Version: 1.26
Status: Development - Tiered Exit Optimization
"""

# Temporary dev hack: Add local Freqtrade root to path for internal imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Original imports follow...
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import talib.abstract as ta
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.persistence import Trade

# Import forward-looking predictive mechanisms
import sys
sys.path.append('/Users/williamgleim/Development/07.08.25/dfai-freqtrade')
from mutantdefi_utils import (
    identify_spanning_extrema,
    calculate_surf_acceleration,
    classify_chop_regime
)

# NEW: Custom SVM Model for RBF Kernel (v1.27 - IFreqaiModel for compatibility)
from freqtrade.freqai.freqai_interface import IFreqaiModel
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # For RBF optimization
from typing import Any  # For method return types

class SurfMultiModel_v1_27(IStrategy):
    """
    SurfMultiModel v1.27: SVM+RBF Integration for Omega Prediction
    
    What's New in v1.27:
    - SVM+RBF Model: Replaces XGBoost with sklearn SVR (RBF kernel) for non-linear Omega prediction
    - Simpler: Inline model class, no extra files
    - RBF Params: C=1.0 (regularization), gamma='scale' (kernel width), epsilon=0.1 (tube)
    - Keeps all v1.26 logic: Multi-scale Omega, tiered exits, etc.
    - Target: Improved non-linear capture of fat tails/asymmetry in Omega
    
    Core: FreqAI now uses SVR(kernel='rbf') to predict &-omega_return
    Why RBF? Handles non-linear market regimes better than trees for your Mandelbrot-inspired features.
    """
    
    # ==================================================================================
    # STRATEGY PARAMETERS
    # ==================================================================================
    
    # FreqAI
    use_freqai = True
    
    # Timeframe
    timeframe = '1m'
    
    # ROI table - AGGRESSIVE for 0.45% daily target (v1.26 optimization)
    minimal_roi = {
        "0": 0.020,    # 2.0% immediate (capture quick wins)
        "15": 0.012,   # 1.2% after 15 minutes
        "45": 0.008,   # 0.8% after 45 minutes
        "90": 0.005    # 0.5% after 90 minutes (realistic per-trade target)
    }
    
    # Stoploss: Keep tight - emergency backup only
    stoploss = -0.02  # 2.0% hard stop
    
    # Trailing stop: DISABLED
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False
    
    # Enable shorts
    can_short = True
    
    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    
    # Other settings
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # ==================================================================================
    # OMEGA PARAMETERS (Hyperopt-optimizable)
    # ==================================================================================
    
    # Omega calculation (same as v1.22)
    omega_forward_window = IntParameter(8, 20, default=12, space='buy', optimize=True)
    omega_threshold = DecimalParameter(-0.002, 0.002, default=0.0, space='buy', optimize=True)
    
    # Multi-scale Omega windows (NEW for v1.23) - Optimized for 1m timeframe
    # Short = omega_forward_window (12 bars = 12 min immediate quality)
    omega_medium_window = IntParameter(20, 40, default=30, space='buy', optimize=True)    # 30 min short-term
    omega_long_window = IntParameter(50, 90, default=60, space='buy', optimize=True)      # 60 min medium-term
    
    # Entry thresholds - Adaptive for FreqAI availability
    # Without FreqAI: 8.0 (technical only), With FreqAI: 12.0 (quality filter)
    technical_score_threshold = DecimalParameter(6.0, 15.0, default=8.0, space='buy', optimize=True)
    
    # Scale consistency threshold (NEW)
    # High consistency = simple/predictable market
    min_scale_consistency = DecimalParameter(0.2, 0.6, default=0.3, space='buy', optimize=True)
    
    # Position sizing
    min_position_multiplier = DecimalParameter(0.2, 0.5, default=0.3, space='buy', optimize=True)
    max_position_multiplier = DecimalParameter(2.0, 3.0, default=2.5, space='buy', optimize=True)
    
    # Directional position limiting (v1.26 - Risk management)
    max_directional_ratio = DecimalParameter(0.5, 0.8, default=0.6, space='buy', optimize=True)
    
    # SVM RBF Hyperopt Params (NEW for v1.27 tuning)
    svm_C = DecimalParameter(0.1, 10.0, default=1.0, space='buy', optimize=True)
    svm_gamma = CategoricalParameter(['scale', 'auto', 0.001, 0.01, 0.1, 1.0], default='scale', space='buy', optimize=True)
    svm_epsilon = DecimalParameter(0.01, 0.5, default=0.1, space='buy', optimize=True)
    
    # ==================================================================================
    # EXIT PARAMETERS (v1.26 - Hyperopt-optimizable)
    # ==================================================================================
    
    # Emergency exit thresholds (populate_exit_trend) - Direction-specific
    emergency_omega_threshold = DecimalParameter(0.15, 0.30, default=0.25, space='sell', optimize=True)
    emergency_accel_threshold = DecimalParameter(0.010, 0.020, default=0.012, space='sell', optimize=True)
    
    # Profit reversal thresholds (populate_exit_trend)
    reversal_omega_pred_threshold = DecimalParameter(0.03, 0.08, default=0.05, space='sell', optimize=True)
    reversal_omega_realized_threshold = DecimalParameter(0.08, 0.15, default=0.10, space='sell', optimize=True)
    reversal_accel_threshold = DecimalParameter(0.008, 0.015, default=0.010, space='sell', optimize=True)
    
    # Tiered exit thresholds (custom_exit)
    # TIER 0: Profit protection (new)
    profit_protect_threshold = DecimalParameter(0.003, 0.010, default=0.005, space='sell', optimize=True)
    profit_protect_omega = DecimalParameter(-0.25, -0.15, default=-0.20, space='sell', optimize=True)
    profit_protect_accel = DecimalParameter(-0.020, -0.010, default=-0.015, space='sell', optimize=True)
    
    # TIER 1: Deep loss
    tier1_loss_threshold = DecimalParameter(-0.015, -0.008, default=-0.010, space='sell', optimize=True)
    tier1_omega_threshold = DecimalParameter(-0.20, -0.10, default=-0.15, space='sell', optimize=True)
    
    # TIER 2: Moderate loss
    tier2_loss_min = DecimalParameter(-0.012, -0.008, default=-0.010, space='sell', optimize=True)
    tier2_loss_max = DecimalParameter(-0.007, -0.004, default=-0.005, space='sell', optimize=True)
    tier2_omega_threshold = DecimalParameter(-0.25, -0.15, default=-0.20, space='sell', optimize=True)
    tier2_accel_threshold = DecimalParameter(-0.018, -0.008, default=-0.012, space='sell', optimize=True)
    
    # TIER 3: Small loss
    tier3_loss_min = DecimalParameter(-0.007, -0.004, default=-0.005, space='sell', optimize=True)
    tier3_loss_max = DecimalParameter(-0.003, -0.001, default=-0.002, space='sell', optimize=True)
    tier3_omega_threshold = DecimalParameter(-0.30, -0.20, default=-0.25, space='sell', optimize=True)
    tier3_accel_threshold = DecimalParameter(-0.020, -0.010, default=-0.015, space='sell', optimize=True)
    tier3_consistency_threshold = DecimalParameter(0.15, 0.25, default=0.20, space='sell', optimize=True)
    
    # ==================================================================================
    # OMEGA CALCULATION (from v1.22)
    # ==================================================================================
    
    def calculate_omega_realized(
        self,
        df: pd.DataFrame,
        lookback_window: int = 12,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Calculate REALIZED Omega ratio (backward-looking) for live trading approximation.
        
        This is a proxy for FreqAI's predicted Omega when model isn't trained yet.
        Uses past returns instead of future returns.
        
        Args:
            df: Price dataframe with 'close' column
            lookback_window: Historical period (# candles to look back)
            threshold: Return threshold for gain/loss calculation
        
        Returns:
            Realized Omega ratio, normalized to [-1, 1] range via tanh
        """
        # Calculate historical returns (backward-looking)
        historical_returns = df['close'].pct_change(periods=lookback_window)
        
        # Rolling Omega calculation
        omega_values = []
        
        for i in range(len(df)):
            # Check if we have enough historical data
            if i < lookback_window:
                omega_values.append(0.0)  # Not enough history
                continue
            
            # Get past returns in the lookback window
            past_returns = historical_returns.iloc[i-lookback_window:i]
            
            # Separate gains and losses relative to threshold
            gains = past_returns[past_returns > threshold] - threshold
            losses = threshold - past_returns[past_returns < threshold]
            
            # Calculate sums
            sum_gains = gains.sum() if len(gains) > 0 else 0
            sum_losses = losses.sum() if len(losses) > 0 else 0.0001
            
            # Omega ratio
            if sum_losses > 0:
                omega = sum_gains / sum_losses
            else:
                omega = 0
            
            # Normalize: Omega=1.0 is neutral, >1.0 = more gains, <1.0 = more losses
            # Convert to [-1, 1] range: Omega-1.0 then tanh
            omega_normalized = np.tanh(omega - 1.0)
            
            omega_values.append(omega_normalized)
        
        return pd.Series(omega_values, index=df.index)
    
    def calculate_omega_target(
        self, 
        df: pd.DataFrame, 
        forward_window: int = 12, 
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Calculate forward-looking Omega ratio as prediction target.
        
        Omega Ratio = Sum(gains above threshold) / Sum(losses below threshold)
        
        This captures Mandelbrot's key insights:
        - Fat tails: SUMS (not variance) give full weight to extremes
        - Asymmetry: Separate gains/losses (no symmetry assumption)
        - Non-Gaussian: Distribution-free (works for any shape)
        
        Args:
            df: Price dataframe with 'close' column
            forward_window: Lookahead period (# candles)
            threshold: Return threshold for gain/loss calculation (default: 0 = breakeven)
        
        Returns:
            Omega ratio series, normalized to [-1, 1] range via tanh
        """
        # Calculate forward returns
        forward_returns = df['close'].pct_change(periods=forward_window).shift(-forward_window)
        
        # Rolling Omega calculation
        omega_values = []
        
        for i in range(len(df)):
            # Check if we have enough future data
            if i + forward_window >= len(df):
                omega_values.append(np.nan)
                continue
            
            # Get future returns in the forward window
            future_returns = forward_returns.iloc[i:i+forward_window]
            
            # Separate gains and losses relative to threshold
            gains = future_returns[future_returns > threshold] - threshold
            losses = threshold - future_returns[future_returns < threshold]
            
            # Calculate sums
            sum_gains = gains.sum() if len(gains) > 0 else 0
            sum_losses = losses.sum() if len(losses) > 0 else 0.0001  # Avoid division by zero
            
            # Omega ratio
            if sum_losses > 0:
                omega = sum_gains / sum_losses
            else:
                omega = 0
            
            # Normalize: Omega=1.0 is neutral (equal gains/losses)
            # >1.0 = more gains, <1.0 = more losses
            # Convert to [-1, 1] range: Omega-1.0 then tanh
            omega_normalized = np.tanh(omega - 1.0)
            
            omega_values.append(omega_normalized)
        
        return pd.Series(omega_values, index=df.index)
    
    # ==================================================================================
    # MULTI-SCALE OMEGA (NEW for v1.23 - Mandelbrot Analysis)
    # ==================================================================================
    
    def calculate_multi_scale_omega(
        self, 
        dataframe: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Calculate Omega at multiple time scales to reveal Mandelbrot properties.
        
        Multi-scale Omega reveals:
        1. Scale consistency (self-similarity / fractal simplicity)
        2. Trend strength (quality trajectory: improving vs degrading)
        3. Market regime (trending vs mean-reverting vs choppy)
        
        This is Mandelbrot's key insight: Markets are SCALE INVARIANT
        - If Omega is consistent across scales → Simple, predictable
        - If Omega diverges across scales → Complex, fractal, unpredictable
        
        Returns:
            Dict with keys:
                - omega_short: 12-bar Omega (immediate quality)
                - omega_medium: 60-bar Omega (1-hour quality)
                - omega_long: 240-bar Omega (4-hour quality)
                - scale_consistency: 0-1 (high = simple/predictable)
                - trend_strength: Ratio (>1 = improving, <1 = degrading)
        """
        # Calculate Omega at 3 time scales
        # Use REALIZED (backward-looking) Omega for live trading approximation
        # This provides a proxy until FreqAI trains
        omega_short = self.calculate_omega_realized(
            dataframe, 
            lookback_window=self.omega_forward_window.value,
            threshold=self.omega_threshold.value
        )
        
        omega_medium = self.calculate_omega_realized(
            dataframe,
            lookback_window=self.omega_medium_window.value,
            threshold=self.omega_threshold.value
        )
        
        omega_long = self.calculate_omega_realized(
            dataframe,
            lookback_window=self.omega_long_window.value,
            threshold=self.omega_threshold.value
        )
        
        # ========================================================================
        # SCALE CONSISTENCY (Mandelbrot's Self-Similarity / Fractal Simplicity)
        # ========================================================================
        # If Omega is similar across scales → Market is self-similar (fractal but simple)
        # If Omega diverges across scales → Market is complex/unpredictable
        
        # Calculate scale ratios
        ratio_sm = (omega_short + 1) / (omega_medium + 1 + 1e-8)  # Add 1 to handle negative Omegas
        ratio_ml = (omega_medium + 1) / (omega_long + 1 + 1e-8)
        
        # Fractal complexity = deviation from 1.0 (perfect self-similarity)
        # Use log to treat ratio=2.0 same as ratio=0.5 (symmetric)
        complexity = np.abs(np.log(ratio_sm + 1e-8)) + np.abs(np.log(ratio_ml + 1e-8))
        
        # Scale consistency = inverse of complexity (0-1 range)
        # High consistency = simple/predictable, Low = complex/unpredictable
        scale_consistency = 1.0 / (complexity + 1)
        scale_consistency = pd.Series(scale_consistency, index=dataframe.index)
        
        # ========================================================================
        # TREND STRENGTH (Quality Trajectory: Improving vs Degrading)
        # ========================================================================
        # Ratio of long-term to short-term Omega
        # > 1.0: Quality improving over time (trending regime)
        # < 1.0: Quality degrading over time (mean-reverting or choppy)
        
        trend_strength = (omega_long + 1) / (omega_short + 1 + 1e-8)
        trend_strength = pd.Series(trend_strength, index=dataframe.index)
        
        return {
            'omega_short': omega_short,
            'omega_medium': omega_medium,
            'omega_long': omega_long,
            'scale_consistency': scale_consistency,
            'trend_strength': trend_strength
        }
    
    # ==================================================================================
    # FREQAI INTEGRATION (same as v1.22 - no changes needed!)
    # ==================================================================================
    
    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define prediction targets for FreqAI.
        
        Primary target: Omega ratio (same as v1.22)
        
        NOTE: We use the SAME Omega target as v1.22!
        The multi-scale analysis happens in populate_indicators()
        FreqAI still predicts single Omega (short-term window)
        """
        # Main target: Omega ratio (short-term)
        dataframe["&-omega_return"] = self.calculate_omega_target(
            dataframe,
            forward_window=self.omega_forward_window.value,
            threshold=self.omega_threshold.value
        )
        
        # Optional: Also predict simple return for comparison/validation
        dataframe["&-simple_return"] = (
            dataframe['close']
            .pct_change(periods=self.omega_forward_window.value)
            .shift(-self.omega_forward_window.value)
        )
        
        return dataframe
    
    def feature_engineering_expand_all(
        self, 
        dataframe: pd.DataFrame, 
        period: int,
        metadata: dict,
        **kwargs
    ) -> pd.DataFrame:
        """
        Feature engineering: Standard indicators across multiple timeframes.
        (Same as v1.22)
        """
        # Basic price features
        dataframe[f"%-roc_{period}"] = ta.ROC(dataframe['close'], timeperiod=period)
        dataframe[f"%-rsi_{period}"] = ta.RSI(dataframe['close'], timeperiod=period)
        
        # Volatility
        dataframe[f"%-atr_{period}"] = ta.ATR(
            dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=period
        )
        dataframe[f"%-volatility_{period}"] = (
            dataframe['close'].pct_change().rolling(period).std()
        )
        
        # Volume
        dataframe[f"%-volume_ratio_{period}"] = (
            dataframe['volume'] / dataframe['volume'].rolling(period).mean()
        )
        
        # Moving averages
        dataframe[f"%-ema_{period}"] = ta.EMA(dataframe['close'], timeperiod=period)
        dataframe[f"%-sma_{period}"] = ta.SMA(dataframe['close'], timeperiod=period)
        
        return dataframe
    
    def feature_engineering_expand_basic(
        self, 
        dataframe: pd.DataFrame, 
        metadata: dict,
        **kwargs
    ) -> pd.DataFrame:
        """
        Additional features (computed once, not per period).
        
        Focus on ASYMMETRY DETECTORS for Omega prediction.
        (Same as v1.22)
        """
        # ============================================================================
        # ASYMMETRY FEATURE SET 1: Realized Skewness
        # ============================================================================
        returns = dataframe['close'].pct_change()
        
        for period in [5, 10, 20, 50]:
            dataframe[f"%-skew_{period}"] = returns.rolling(period).skew()
        
        # ============================================================================
        # ASYMMETRY FEATURE SET 2: Upside/Downside Volatility Ratio
        # ============================================================================
        for period in [10, 20, 50]:
            upside_vol = returns.where(returns > 0, 0).rolling(period).std()
            downside_vol = returns.where(returns < 0, 0).rolling(period).std()
            
            dataframe[f"%-vol_asymmetry_{period}"] = (
                upside_vol / (downside_vol + 1e-8)
            )
        
        # ============================================================================
        # ASYMMETRY FEATURE SET 3: Tail Frequency (>2σ moves)
        # ============================================================================
        for period in [20, 50]:
            std = returns.rolling(period).std()
            
            upside_tail = (returns > 2 * std).rolling(period).mean()
            downside_tail = (returns < -2 * std).rolling(period).mean()
            
            dataframe[f"%-tail_ratio_{period}"] = (
                upside_tail / (downside_tail + 1e-8)
            )
        
        # ============================================================================
        # ASYMMETRY FEATURE SET 4: Volume-Weighted Directional Bias
        # ============================================================================
        volume_pct = dataframe['volume'] / dataframe['volume'].rolling(20).mean()
        
        for period in [10, 20]:
            # Volume-weighted returns
            vol_weighted_returns = returns * volume_pct
            
            upside_vol_weighted = (
                vol_weighted_returns.where(returns > 0, 0).rolling(period).sum()
            )
            downside_vol_weighted = (
                vol_weighted_returns.where(returns < 0, 0).rolling(period).sum()
            )
            
            dataframe[f"%-vol_directional_{period}"] = (
                upside_vol_weighted / (abs(downside_vol_weighted) + 1e-8)
            )
        
        # ============================================================================
        # ASYMMETRY FEATURE SET 5: Drawdown/Runup Ratio
        # ============================================================================
        for period in [20, 50]:
            # Maximum runup (gain from trough)
            roll_min = dataframe['close'].rolling(period).min()
            runup = (dataframe['close'] - roll_min) / (roll_min + 1e-8)
            
            # Maximum drawdown (loss from peak)
            roll_max = dataframe['close'].rolling(period).max()
            drawdown = (roll_max - dataframe['close']) / (roll_max + 1e-8)
            
            dataframe[f"%-runup_drawdown_ratio_{period}"] = (
                runup / (drawdown + 1e-8)
            )
        
        # ADX (trend strength)
        dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'])
        
        return dataframe
    
    def feature_engineering_standard(
        self, 
        dataframe: pd.DataFrame, 
        metadata: dict,
        **kwargs
    ) -> pd.DataFrame:
        """
        Standard feature transformations (optional).
        """
        return dataframe
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Forward-looking predictive mechanisms + Multi-Scale Omega (NEW for v1.23)
        
        v1.22: Single Omega + forward-looking indicators
        v1.23: Multi-scale Omega (Mandelbrot analysis) + forward-looking indicators
        """
        # =================================================================================
        # 1. REGIME DETECTION: Classify market volatility (from v1.22)
        # =================================================================================
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        chop_ratio, chop_regime = classify_chop_regime(dataframe['atr'])
        dataframe['chop_ratio'] = chop_ratio
        dataframe['chop_regime'] = chop_regime
        
        regime_map = {'heavy_chop': 3, 'moderate_chop': 2, 'light_chop': 1}
        dataframe['regime_score'] = dataframe['chop_regime'].map(regime_map).fillna(2)
        
        # =================================================================================
        # 2. MOMENTUM ACCELERATION (from v1.22)
        # =================================================================================
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['surf_velocity'] = dataframe['returns'].rolling(window=5).mean()
        dataframe['surf_acceleration'] = calculate_surf_acceleration(dataframe['surf_velocity'])
        
        # =================================================================================
        # 3. EXTREMA-SPANNING EXTREMA (from v1.22)
        # =================================================================================
        spanning_maxima, spanning_minima, local_peaks, local_troughs = identify_spanning_extrema(
            dataframe['close'], 
            window=20, 
            significance_threshold=0.10
        )
        dataframe['spanning_max'] = spanning_maxima
        dataframe['spanning_min'] = spanning_minima
        dataframe['local_peak'] = local_peaks
        dataframe['local_trough'] = local_troughs
        
        # Distance from last extrema
        dataframe['bars_since_peak'] = (~dataframe['spanning_max']).cumsum() - (~dataframe['spanning_max']).cumsum().where(dataframe['spanning_max']).ffill().fillna(0)
        dataframe['bars_since_trough'] = (~dataframe['spanning_min']).cumsum() - (~dataframe['spanning_min']).cumsum().where(dataframe['spanning_min']).ffill().fillna(0)
        
        # =================================================================================
        # 4. SIMPLE PRICE ACTION (from v1.22)
        # =================================================================================
        dataframe['high_20'] = dataframe['high'].rolling(window=20).max()
        dataframe['low_20'] = dataframe['low'].rolling(window=20).min()
        dataframe['price_position'] = (dataframe['close'] - dataframe['low_20']) / (dataframe['high_20'] - dataframe['low_20'])
        dataframe['price_position'] = dataframe['price_position'].fillna(0.5)
        
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # =================================================================================
        # 5. MULTI-SCALE OMEGA (NEW for v1.23 - Mandelbrot Analysis!)
        # =================================================================================
        logger.info(f"[{metadata['pair']}] Calculating multi-scale Omega (Mandelbrot analysis)...")
        
        ms_omega = self.calculate_multi_scale_omega(dataframe)
        
        dataframe['omega_short'] = ms_omega['omega_short']
        dataframe['omega_medium'] = ms_omega['omega_medium']
        dataframe['omega_long'] = ms_omega['omega_long']
        dataframe['scale_consistency'] = ms_omega['scale_consistency']
        dataframe['trend_strength'] = ms_omega['trend_strength']
        
        logger.info(f"[{metadata['pair']}] Multi-scale Omega calculated - "
                   f"Consistency range: [{dataframe['scale_consistency'].min():.2f}, {dataframe['scale_consistency'].max():.2f}], "
                   f"Trend strength range: [{dataframe['trend_strength'].min():.2f}, {dataframe['trend_strength'].max():.2f}]")
        
        return dataframe
    
    # ==================================================================================
    # ENTRY LOGIC: Omega-Mandelbrot Quality Scoring (ENHANCED from v1.22)
    # ==================================================================================
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Quality-first entry system (v1.23 Omega-Mandelbrot)
        
        Changes from v1.22:
        - Layer 2: Omega Quality Score (0-15 pts) vs Omega Boost (0-5 pts)
          - Component 1: omega_short (immediate quality) → 0-7 pts
          - Component 2: scale_consistency (Mandelbrot simplicity) → 0-5 pts
          - Component 3: trend_strength (quality trajectory) → 0-3 pts
        - Higher threshold: 12.0 (vs 8.0 in v1.22)
        - More selective: Only high-quality extrema
        """
        # Initialize
        dataframe['enter_long'] = False
        dataframe['enter_short'] = False
        dataframe['enter_tag'] = ''
        
        if len(dataframe) == 0:
            return dataframe
        
        # Get Omega prediction from FreqAI (default to 0 if unavailable)
        omega_pred = dataframe.get('&-omega_return_pred', pd.Series(0, index=dataframe.index)).fillna(0)
        
        # Get multi-scale Omega metrics with cascading fallback
        # Cascade: long → medium → short → 0
        # This uses the closest-available timeframe when longer ones aren't ready yet
        omega_short = dataframe['omega_short'].fillna(0)
        omega_medium = dataframe['omega_medium'].fillna(omega_short)  # Cascade: medium → short
        omega_long = dataframe['omega_long'].fillna(omega_medium).fillna(omega_short)  # Cascade: long → medium → short
        
        # Scale consistency: If NaN, assume moderate (0.5)
        scale_consistency = dataframe['scale_consistency'].fillna(0.5)
        
        # Trend strength: If NaN, assume neutral (1.0)
        trend_strength = dataframe['trend_strength'].fillna(1.0)
        
        # ============================================================================
        # LAYER 1: FORWARD-LOOKING TECHNICAL SCORE (0-10 points, same as v1.22)
        # ============================================================================
        technical_score_long = pd.Series(0.0, index=dataframe.index)
        technical_score_short = pd.Series(0.0, index=dataframe.index)
        
        # Get forward-looking indicators
        price_pos = dataframe['price_position'].fillna(0.5)
        surf_accel = dataframe['surf_acceleration'].fillna(0)
        regime_score = dataframe['regime_score'].fillna(2)
        bars_since_peak = dataframe['bars_since_peak'].fillna(999)
        bars_since_trough = dataframe['bars_since_trough'].fillna(999)
        
        # Component 1: Extrema Distance (0-4 points)
        extrema_score_long = pd.Series(
            np.where(bars_since_trough <= 1, 4,
            np.where(bars_since_trough <= 2, 2, 0)),
            index=dataframe.index
        )
        technical_score_long += extrema_score_long
        
        extrema_score_short = pd.Series(
            np.where(bars_since_peak <= 1, 4,
            np.where(bars_since_peak <= 2, 2, 0)),
            index=dataframe.index
        )
        technical_score_short += extrema_score_short
        
        # Component 2: Momentum Acceleration (0-4 points)
        accel_score_long = (surf_accel.clip(0, 0.05) * 80).clip(0, 4)
        technical_score_long += accel_score_long
        
        accel_score_short = ((-surf_accel).clip(0, 0.05) * 80).clip(0, 4)
        technical_score_short += accel_score_short
        
        # Component 3: Regime Adaptation (0-2 points)
        regime_bonus_long = pd.Series(
            np.where(regime_score == 1, 2, 0),
            index=dataframe.index
        )
        technical_score_long += regime_bonus_long
        technical_score_short += regime_bonus_long
        
        # ============================================================================
        # LAYER 2: OMEGA QUALITY SCORE (0-15 points, NEW for v1.23)
        # ============================================================================
        # This replaces v1.22's "Omega Boost" (0-5 pts) with enhanced scoring
        
        # Component 1: Immediate Quality (omega_short) → 0-7 points
        # Adaptive Omega: FreqAI (best) → Realized Omega (proxy) → 0 (fallback)
        # This allows trading before FreqAI trains, with gradual quality improvement
        omega_short_safe = omega_short.fillna(0)  # Extra safety
        
        # Use FreqAI if available (most accurate), else use realized Omega (proxy)
        omega_for_scoring = np.where(
            np.abs(omega_pred) > 0.01,   # If FreqAI has confident prediction
            omega_pred,                   # Use it (best quality)
            omega_short_safe * 0.7        # Else use realized Omega with 30% discount (proxy)
        )
        omega_quality_long = (pd.Series(omega_for_scoring, index=dataframe.index).clip(0, 1) * 7).clip(0, 7)
        omega_quality_short = ((-pd.Series(omega_for_scoring, index=dataframe.index)).clip(0, 1) * 7).clip(0, 7)
        
        # Component 2: Scale Consistency (Mandelbrot simplicity) → 0-5 points
        # High consistency = simple/predictable market = boost entry
        consistency_score = (scale_consistency * 5).clip(0, 5)
        
        # Component 3: Trend Strength (quality trajectory) → 0-3 points
        # For longs: trend_strength > 1.0 is good (improving quality)
        # For shorts: trend_strength < 1.0 is good (degrading quality)
        # CRITICAL: If trend_strength is NaN or 1.0 (neutral), score is 0
        trend_strength_safe = trend_strength.fillna(1.0)  # NaN → neutral
        trend_score_long = ((trend_strength_safe - 1.0).clip(0, 0.5) * 6).clip(0, 3)
        trend_score_short = ((1.0 - trend_strength_safe).clip(0, 0.5) * 6).clip(0, 3)
        
        # Total Omega Quality Score
        quality_score_long = omega_quality_long + consistency_score + trend_score_long
        quality_score_short = omega_quality_short + consistency_score + trend_score_short
        
        # ============================================================================
        # COMBINED ENTRY SCORES
        # ============================================================================
        long_score = technical_score_long + quality_score_long
        short_score = technical_score_short + quality_score_short
        
        # Store scores in dataframe for exit logic to use
        dataframe['long_score'] = long_score
        dataframe['short_score'] = short_score
        
        # ============================================================================
        # ENTRY CONDITIONS (Quality-First + Extrema Competition)
        # ============================================================================
        # Threshold: 12.0 (higher than v1.22's 8.0)
        # Rationale: We want HIGH QUALITY trades only
        
        volume_ok = dataframe['volume'] > 0
        
        # Scale consistency filter (Mandelbrot filter)
        # Reject entries in complex/unpredictable markets
        scale_ok = scale_consistency >= self.min_scale_consistency.value
        
        # Extrema competition: Only enter at BEST local extrema (tighten from v1.22)
        # Longs prefer troughs (buy low), shorts prefer peaks (sell high)
        extrema_competition_long = (
            dataframe['bars_since_trough'] <= 1  # Only at immediate trough (not <=3)
        )
        extrema_competition_short = (
            dataframe['bars_since_peak'] <= 1  # Only at immediate peak (not <=3)
        )
        
        # Long entry
        long_conditions = (
            volume_ok &
            scale_ok &  # NEW: Scale consistency filter
            (long_score >= self.technical_score_threshold.value) &
            extrema_competition_long
        )
        
        # Short entry
        short_conditions = (
            volume_ok &
            scale_ok &  # NEW: Scale consistency filter
            (short_score >= self.technical_score_threshold.value) &
            extrema_competition_short
        )
        
        # Mutual exclusivity
        dataframe.loc[long_conditions & ~short_conditions, 'enter_long'] = True
        dataframe.loc[short_conditions & ~long_conditions, 'enter_short'] = True
        
        # Tags
        dataframe.loc[dataframe['enter_long'], 'enter_tag'] = 'omega_mandelbrot_long'
        dataframe.loc[dataframe['enter_short'], 'enter_tag'] = 'omega_mandelbrot_short'
        
        # Enhanced debug logging (last candle only)
        if len(dataframe) > 0:
            last_idx = dataframe.index[-1]
            
            def get_last(val):
                return val.iloc[-1] if hasattr(val, 'iloc') else (val[-1] if hasattr(val, '__getitem__') else val)
            
            # Check for NaN fallbacks
            omega_medium_raw = get_last(dataframe['omega_medium'])
            omega_long_raw = get_last(dataframe['omega_long'])
            using_fallback = pd.isna(omega_medium_raw) or pd.isna(omega_long_raw)
            
            logger.info(f"""
                === v1.23 Omega-Mandelbrot Entry Analysis [{metadata['pair']}] ===
                
                MULTI-SCALE OMEGA (Mandelbrot):
                  Omega Short ({self.omega_forward_window.value}-bar):  {get_last(omega_short):+.3f} (immediate quality)
                  Omega Medium ({self.omega_medium_window.value}-bar): {get_last(omega_medium):+.3f} (short-term){' [FALLBACK]' if pd.isna(omega_medium_raw) else ''}
                  Omega Long ({self.omega_long_window.value}-bar):  {get_last(omega_long):+.3f} (medium-term){' [FALLBACK]' if pd.isna(omega_long_raw) else ''}
                  
                MANDELBROT METRICS:
                  Scale Consistency: {get_last(scale_consistency):.3f} (0-1, high = simple/predictable)
                  Trend Strength:    {get_last(trend_strength):.3f} (>1 = improving, <1 = degrading)
                  Scale Filter:      {'✓ PASS' if get_last(scale_ok) else '✗ FAIL (too complex)'}
                
                EXTREMA COMPETITION:
                  Bars Since Trough: {get_last(bars_since_trough):.0f} (0-1 = BEST, ≤3 = ELIGIBLE)
                  Bars Since Peak:   {get_last(bars_since_peak):.0f} (0-1 = BEST, ≤3 = ELIGIBLE)
                
                SCORE BREAKDOWN (Long):
                  Technical (0-10):      {get_last(technical_score_long):.1f}
                  Quality (0-15):        {get_last(quality_score_long):.1f}
                    ├─ Omega:            {get_last(omega_quality_long):.1f} / 7.0
                    ├─ Consistency:      {get_last(consistency_score):.1f} / 5.0
                    └─ Trend:            {get_last(trend_score_long):.1f} / 3.0
                  → TOTAL:               {get_last(long_score):.1f} / {self.technical_score_threshold.value:.1f}
                
                SCORE BREAKDOWN (Short):
                  Technical (0-10):      {get_last(technical_score_short):.1f}
                  Quality (0-15):        {get_last(quality_score_short):.1f}
                    ├─ Omega:            {get_last(omega_quality_short):.1f} / 7.0
                    ├─ Consistency:      {get_last(consistency_score):.1f} / 5.0
                    └─ Trend:            {get_last(trend_score_short):.1f} / 3.0
                  → TOTAL:               {get_last(short_score):.1f} / {self.technical_score_threshold.value:.1f}
                
                ENTRY SIGNALS:
                  Long:  {'✓ ENTER (high quality!)' if dataframe['enter_long'].iloc[-1] else '✗ NO'}
                  Short: {'✓ ENTER (high quality!)' if dataframe['enter_short'].iloc[-1] else '✗ NO'}
                =====================================================================
            """)
        
        return dataframe
    
    # ==================================================================================
    # EXIT LOGIC: Scale-Aware Exits (ENHANCED from v1.22)
    # ==================================================================================
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        v1.26: QUALITY-MIRRORED EXIT LOGIC - Aligned with Entry Scoring
        
        Philosophy:
        - Exit when quality score drops below entry threshold (mirrored logic)
        - Calculate exit quality scores using same framework as entry
        - Long exits at peaks with SHORT quality score (for short entry)
        - Short exits at troughs with LONG quality score (for long entry)
        
        Target: 2.5-3.0x exit/entry ratio with quality-based exits
        """
        dataframe['exit_long'] = False
        dataframe['exit_short'] = False
        dataframe['exit_tag'] = ''
        
        if len(dataframe) == 0:
            return dataframe
        
        # Get indicators (already calculated in populate_indicators)
        omega_short = dataframe['omega_short'].fillna(0)
        omega_medium = dataframe['omega_medium'].fillna(0)
        omega_long = dataframe['omega_long'].fillna(0)
        scale_consistency = dataframe['scale_consistency'].fillna(0.5)
        trend_strength = dataframe['trend_strength'].fillna(1.0)
        bars_since_peak = dataframe['bars_since_peak'].fillna(999)
        bars_since_trough = dataframe['bars_since_trough'].fillna(999)
        volume_ok = dataframe['volume'] > 0
        
        # Get quality scores (already calculated in populate_entry_trend)
        long_score = dataframe.get('long_score', pd.Series(0, index=dataframe.index)).fillna(0)
        short_score = dataframe.get('short_score', pd.Series(0, index=dataframe.index)).fillna(0)
        
        # =========================================================================
        # LONG EXITS: Exit at peaks when SHORT quality is HIGH (time to reverse)
        # =========================================================================
        # Logic: At peaks, if conditions favor SHORT entry, exit LONG
        # This mirrors the entry system: exit when opposite direction has good setup
        exit_long = (
            (bars_since_peak <= 3) &  # At or near peak (sell high)
            (
                # Exit if SHORT quality score is good (opposite direction favored)
                (short_score >= self.technical_score_threshold.value * 0.7) |  # 70% of entry threshold
                # OR if omega has flipped negative (quality degraded)
                (omega_short < -0.05) |
                # OR if both momentum and complexity have degraded
                ((dataframe['surf_acceleration'].fillna(0) < -0.008) & (scale_consistency < 0.30))
            )
        ) | (
            # OR exit on SEVERE quality collapse anywhere
            (omega_short < -self.emergency_omega_threshold.value) &
            (dataframe['surf_acceleration'].fillna(0) < -self.emergency_accel_threshold.value) &
            (scale_consistency < 0.20)
        )
        
        # =========================================================================
        # SHORT EXITS: Exit at troughs when LONG quality is HIGH (time to reverse)
        # =========================================================================
        # Logic: At troughs, if conditions favor LONG entry, exit SHORT
        # This mirrors the entry system: exit when opposite direction has good setup
        exit_short = (
            (bars_since_trough <= 3) &  # At or near trough (buy back low)
            (
                # Exit if LONG quality score is good (opposite direction favored)
                (long_score >= self.technical_score_threshold.value * 0.7) |  # 70% of entry threshold
                # OR if omega has flipped positive (quality degraded)
                (omega_short > 0.05) |
                # OR if both momentum and complexity have degraded
                ((dataframe['surf_acceleration'].fillna(0) > 0.008) & (scale_consistency < 0.30))
            )
        ) | (
            # OR exit on SEVERE quality collapse anywhere
            (omega_short > self.emergency_omega_threshold.value) &
            (dataframe['surf_acceleration'].fillna(0) > self.emergency_accel_threshold.value) &
            (scale_consistency < 0.20)
        )
        
        # =========================================================================
        # Apply exits with tags
        # =========================================================================
        dataframe.loc[volume_ok & exit_long, 'exit_long'] = True
        dataframe.loc[volume_ok & exit_long & (bars_since_peak <= 3), 'exit_tag'] = 'profit_reversal'
        dataframe.loc[volume_ok & exit_long & (bars_since_peak > 3), 'exit_tag'] = 'quality_collapse'
        
        dataframe.loc[volume_ok & exit_short, 'exit_short'] = True
        dataframe.loc[volume_ok & exit_short & (bars_since_trough <= 3), 'exit_tag'] = 'profit_reversal'
        dataframe.loc[volume_ok & exit_short & (bars_since_trough > 3), 'exit_tag'] = 'quality_collapse'
        
        return dataframe
    
    # ==================================================================================
    # DYNAMIC POSITION SIZING: Multi-Factor Omega (ENHANCED from v1.22)
    # ==================================================================================
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Multi-factor position sizing based on Omega-Mandelbrot metrics.
        
        v1.22: Single-factor (Omega magnitude)
        v1.23: Multi-factor (Omega × consistency × trend)
        
        Logic:
        - Omega multiplier: 0.5x to 2.0x based on immediate quality
        - Consistency adjustment: 0.8x to 1.2x (simple market = boost)
        - Trend adjustment: 0.8x to 1.2x (improving quality = boost)
        - Final range: 0.3x to 2.5x
        """
        # Get latest dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) == 0:
            logger.warning(f"No dataframe for {pair}, using base stake")
            return proposed_stake
        
        # Get Omega prediction (prefer FreqAI, fallback to calculated)
        if '&-omega_return_pred' in dataframe.columns:
            omega_pred = dataframe['&-omega_return_pred'].iloc[-1]
            if pd.isna(omega_pred):
                omega_pred = dataframe['omega_short'].iloc[-1] if 'omega_short' in dataframe.columns else 0.0
        else:
            omega_pred = dataframe['omega_short'].iloc[-1] if 'omega_short' in dataframe.columns else 0.0
        
        # Get multi-scale metrics
        scale_consistency = dataframe['scale_consistency'].iloc[-1] if 'scale_consistency' in dataframe.columns else 0.5
        trend_strength = dataframe['trend_strength'].iloc[-1] if 'trend_strength' in dataframe.columns else 1.0
        
        # ========================================================================
        # Factor 1: Omega Multiplier (immediate quality)
        # ========================================================================
        abs_omega = abs(omega_pred)
        omega_mult = self.min_position_multiplier.value + (
            abs_omega * (self.max_position_multiplier.value - self.min_position_multiplier.value)
        )
        omega_mult = np.clip(omega_mult, self.min_position_multiplier.value, self.max_position_multiplier.value)
        
        # ========================================================================
        # Factor 2: Consistency Adjustment (Mandelbrot simplicity)
        # ========================================================================
        # High consistency = simple/predictable = increase size
        # Range: 0.8x (complex) to 1.2x (simple)
        consistency_adj = 0.8 + (scale_consistency * 0.4)
        consistency_adj = np.clip(consistency_adj, 0.8, 1.2)
        
        # ========================================================================
        # Factor 3: Trend Adjustment (quality trajectory)
        # ========================================================================
        # For longs: trend_strength > 1.2 = improving quality = boost
        # For shorts: trend_strength < 0.8 = degrading quality = boost
        if side == 'long':
            if trend_strength > 1.2:
                trend_adj = 1.2  # Quality improving
            elif trend_strength < 0.8:
                trend_adj = 0.8  # Quality degrading (bad for longs)
            else:
                trend_adj = 1.0
        else:  # short
            if trend_strength < 0.8:
                trend_adj = 1.2  # Quality degrading (good for shorts)
            elif trend_strength > 1.2:
                trend_adj = 0.8  # Quality improving (bad for shorts)
            else:
                trend_adj = 1.0
        
        # ========================================================================
        # Combined Multiplier
        # ========================================================================
        total_mult = omega_mult * consistency_adj * trend_adj
        final_mult = np.clip(total_mult, self.min_position_multiplier.value, self.max_position_multiplier.value)
        
        # Apply multiplier
        adjusted_stake = proposed_stake * final_mult
        
        # Ensure within exchange limits
        if min_stake is not None:
            adjusted_stake = max(min_stake, adjusted_stake)
        adjusted_stake = min(adjusted_stake, max_stake)
        
        # Logging
        logger.info(f"""
            Multi-Factor Position Sizing [{pair}]:
            Omega: {omega_pred:+.3f} → mult {omega_mult:.2f}x
            Scale Consistency: {scale_consistency:.3f} → adj {consistency_adj:.2f}x
            Trend Strength: {trend_strength:.3f} → adj {trend_adj:.2f}x
            → Total Multiplier: {final_mult:.2f}x
            Base Stake: {proposed_stake:.2f}
            Adjusted Stake: {adjusted_stake:.2f}
            (Range: {self.min_position_multiplier.value:.1f}x - {self.max_position_multiplier.value:.1f}x)
        """)
        
        return adjusted_stake
    
    # ==================================================================================
    # CUSTOM EXIT (NEW v1.24.1 - Omega Quality Monitoring)
    # ==================================================================================
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """
        v1.26: TIERED EXIT LOGIC - Profit-Aware Quality Management
        
        Exit tiers (evaluated in order):
        - TIER 0: PROFIT PROTECTION (>= +0.5%) - Exit on severe reversal
        - TIER 1: DEEP LOSS (< -1.0%) - Pre-empt stoploss
        - TIER 2: MODERATE LOSS (-0.5% to -1.0%) - Quality collapse
        - TIER 3: SMALL LOSS (-0.2% to -0.5%) - Extreme collapse only
        - TIER 4: SMALL PROFIT/NEUTRAL (-0.2% to +0.5%) - No exits, let ROI work
        
        Goal: Eliminate 90% of exit signals, keep 10% that matter
        Expected: 280 entries → ~300-350 exits (1.2x ratio)
        """
        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) == 0:
            return None
        
        # Get current Omega metrics
        current_omega = dataframe['omega_short'].iloc[-1] if 'omega_short' in dataframe.columns else 0.0
        current_consistency = dataframe['scale_consistency'].iloc[-1] if 'scale_consistency' in dataframe.columns else 0.5
        surf_accel = dataframe['surf_acceleration'].iloc[-1] if 'surf_acceleration' in dataframe.columns else 0.0
        
        is_long = not trade.is_short
        
        # ========================================================================
        # TIER 0: PROFIT PROTECTION - Catch Reversals in Profitable Territory
        # ========================================================================
        # If we're in profit but quality collapsed + strong reversal, exit
        # This protects +0.5% to +1.5% profits from turning into losses
        if current_profit >= self.profit_protect_threshold.value:
            if is_long:
                # Exit long if Omega flipped negative AND strong downward acceleration
                if current_omega < self.profit_protect_omega.value and surf_accel < self.profit_protect_accel.value:
                    logger.info(f"[{pair}] TIER 0: Profit protection exit (profit: {current_profit:.2%}, omega: {current_omega:.3f}, accel: {surf_accel:.4f})")
                    return "tier0_profit_protect"
            else:
                # Exit short if Omega flipped positive AND strong upward acceleration
                if current_omega > -self.profit_protect_omega.value and surf_accel > -self.profit_protect_accel.value:
                    logger.info(f"[{pair}] TIER 0: Profit protection exit (profit: {current_profit:.2%}, omega: {current_omega:.3f}, accel: {surf_accel:.4f})")
                    return "tier0_profit_protect"
        
        # ========================================================================
        # TIER 1: DEEP LOSS - Pre-empt Stoploss
        # ========================================================================
        # Exit if Omega quality is negative (no recovery expected)
        # This saves ~0.5-1.0% by exiting before -2% stoploss
        if current_profit < self.tier1_loss_threshold.value:
            if is_long and current_omega < self.tier1_omega_threshold.value:
                logger.info(f"[{pair}] TIER 1: Deep loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f})")
                return "tier1_deep_loss"
            elif not is_long and current_omega > -self.tier1_omega_threshold.value:
                logger.info(f"[{pair}] TIER 1: Deep loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f})")
                return "tier1_deep_loss"
        
        # ========================================================================
        # TIER 2: MODERATE LOSS - Quality Collapse
        # ========================================================================
        # Exit only if ALL THREE: Omega collapsed AND momentum reversed AND consistency low
        # This prevents premature exits during temporary chop (high complexity, noise)
        if self.tier2_loss_min.value <= current_profit < self.tier2_loss_max.value:
            if is_long:
                if (current_omega < self.tier2_omega_threshold.value and 
                    surf_accel < self.tier2_accel_threshold.value and
                    current_consistency < 0.25):
                    logger.info(f"[{pair}] TIER 2: Moderate loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f}, consistency: {current_consistency:.2f})")
                    return "tier2_moderate_loss"
            else:
                if (current_omega > -self.tier2_omega_threshold.value and 
                    surf_accel > -self.tier2_accel_threshold.value and
                    current_consistency < 0.25):
                    logger.info(f"[{pair}] TIER 2: Moderate loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f}, consistency: {current_consistency:.2f})")
                    return "tier2_moderate_loss"
        
        # ========================================================================
        # TIER 3: SMALL LOSS - Extreme Collapse Only
        # ========================================================================
        # Exit only if ALL THREE: Omega collapsed + momentum reversed + consistency low
        if self.tier3_loss_min.value <= current_profit < self.tier3_loss_max.value:
            if is_long:
                if (current_omega < self.tier3_omega_threshold.value and 
                    surf_accel < self.tier3_accel_threshold.value and 
                    current_consistency < self.tier3_consistency_threshold.value):
                    logger.info(f"[{pair}] TIER 3: Small loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f})")
                    return "tier3_small_loss"
            else:
                if (current_omega > -self.tier3_omega_threshold.value and 
                    surf_accel > -self.tier3_accel_threshold.value and 
                    current_consistency < self.tier3_consistency_threshold.value):
                    logger.info(f"[{pair}] TIER 3: Small loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f})")
                    return "tier3_small_loss"
        
        # ========================================================================
        # TIER 4: SMALL PROFIT/NEUTRAL (-0.2% to +0.5%) - NO EXITS
        # ========================================================================
        # In this zone, trust the position and let it develop
        # ROI table will handle exits at +2.0%, +1.2%, +0.8%, +0.5%
        # Quality-based exits disabled to avoid overproliferation
        
        return None
    
    # ==================================================================================
    # TRADE CONFIRMATION
    # ==================================================================================
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Final trade entry confirmation with directional position limiting.
        
        v1.26 Risk Management:
        1. Limit 1 position per pair
        2. Prevent majority directional bias (max 60% of pairs in same direction)
        """
        # ========================================================================
        # Check 1: Limit positions per pair (prevent duplicate positions)
        # ========================================================================
        open_trades_for_pair = Trade.get_trades_proxy(pair=pair, is_open=True)
        max_positions_per_pair = 1
        
        if len(open_trades_for_pair) >= max_positions_per_pair:
            logger.info(f"❌ Rejecting {side} for {pair}: Max positions per pair ({max_positions_per_pair}) reached")
            return False
        
        # ========================================================================
        # Check 2: Directional position limiting (prevent correlated risk)
        # ========================================================================
        # Get all open trades across all pairs
        all_open_trades = Trade.get_trades_proxy(is_open=True)
        
        # Count directional positions
        long_count = sum(1 for trade in all_open_trades if not trade.is_short)
        short_count = sum(1 for trade in all_open_trades if trade.is_short)
        total_open = long_count + short_count
        
        # Calculate total possible positions (max_open_trades from config)
        # Use 5 as total pairs since we have 5 in pair_whitelist
        total_pairs = 5  # BTC, ETH, HYPE, SOL, PUMP
        
        # Define directional limit using hyperopt parameter
        # Default 60% means 3 out of 5 pairs max in same direction
        max_directional_positions = int(total_pairs * self.max_directional_ratio.value)
        
        # Check if opening this trade would exceed directional limit
        if side == 'long' and long_count >= max_directional_positions:
            logger.warning(
                f"❌ Rejecting LONG for {pair}: Directional limit reached "
                f"(longs: {long_count}/{max_directional_positions}, shorts: {short_count})"
            )
            return False
        
        if side == 'short' and short_count >= max_directional_positions:
            logger.warning(
                f"❌ Rejecting SHORT for {pair}: Directional limit reached "
                f"(longs: {long_count}, shorts: {short_count}/{max_directional_positions})"
            )
            return False
        
        # ========================================================================
        # All checks passed - confirm entry
        # ========================================================================
        logger.info(
            f"✅ Confirming {side.upper()} entry for {pair} @ {rate} "
            f"(tag: {entry_tag}, positions: L{long_count} S{short_count})"
        )
        return True


# ==================================================================================
# VERSION CONTROL
# ==================================================================================
"""
VERSION: v1.23.0
DATE: October 5, 2025
CHANGELOG:
- Omega-Mandelbrot implementation: Multi-scale Omega analysis
- Added scale_consistency metric (fractal simplicity proxy)
- Added trend_strength metric (quality trajectory)
- Enhanced entry scoring: Omega Quality Score (0-15 pts) vs Omega Boost (0-5 pts)
- Scale-aware exit logic (exit when Mandelbrot properties degrade)
- Multi-factor position sizing (Omega × consistency × trend)
- More selective entries (threshold 12.0 vs 8.0)
- Quality-first trading philosophy

KEY INSIGHT:
Omega ratio IS a Mandelbrot approximator because:
- Fat tails: Captured in SUMS (not variance which underweights)
- Asymmetry: Separate gains/losses (no symmetry assumption)
- Non-Gaussian: Distribution-free (works for any shape)
- Scale invariance: Multi-scale Omega reveals self-similarity

NEXT STEPS:
- Backtest vs v1.22 baseline
- Validate scale_consistency predictive power
- Optimize thresholds (technical_score_threshold, min_scale_consistency)
- Dry run testing
"""

