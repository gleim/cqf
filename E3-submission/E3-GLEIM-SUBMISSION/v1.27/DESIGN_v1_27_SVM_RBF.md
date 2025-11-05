# V1.27 Design: SVM with RBF Kernel for FreqAI Omega Prediction

## Overview
Version 1.27 integrates a Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel into the FreqAI framework for predicting Omega ratio values. This is a minimal, targeted enhancement to v1.26's quality-mirrored exit strategy.

## Core Innovation
- **Model**: `sklearn.svm.SVR` with RBF kernel (`kernel='rbf'`)
- **Target**: Omega ratio (Mandelbrot approximator for fat tails, asymmetry, non-Gaussian behavior)
- **Why RBF?**: Captures non-linear relationships in multi-scale Omega features (12, 60, 240 bars)
- **Integration**: Custom FreqAI model (`SVRSurfModel`) inheriting from `IFreqaiModel`

## Architecture

### 1. Custom FreqAI Model: `SVRSurfModel`
**Location**: `user_data/versions/v1.27/freqaimodels/SVRSurfModel.py`

**Key Features**:
- **Inheritance**: `IFreqaiModel` (abstract base for all FreqAI models)
- **Pipeline**: `StandardScaler` → `SVR(kernel='rbf')`
  - **Scaling**: Critical for RBF (distance-based kernel)
  - **SVR**: Continuous regression for Omega values [-1, 1]
- **Parameters** (from config `model_training_parameters`):
  - `C`: Regularization (default 1.0, hyperopt range 0.1-10.0)
  - `gamma`: Kernel coefficient (default 'scale', hyperopt: 'scale' or 0.001-1.0)
  - `epsilon`: Epsilon-tube for SVR (default 0.1, hyperopt 0.01-0.5)
- **Methods**:
  - `train()`: Full training loop (builds pipeline, fits on train data, saves model)
  - `fit()`: Fit on split data (called after FreqAI data split)
  - `predict()`: Predict Omega on unfiltered_df (returns pred_df, do_predict array)

### 2. Strategy: `SurfMultiModel_v1_27`
**Location**: `user_data/versions/v1.27/strategy/SurfMultiModel_v1_27.py`

**Inheritance**: Copied from v1.26 (quality-mirrored exits, multi-factor sizing)

**Changes from v1.26**:
- **No model class in strategy** (moved to separate `SVRSurfModel.py`)
- **Hyperopt Parameters** (for RBF tuning):
  - `svm_C`: `DecimalParameter(0.1, 10.0, default=1.0, space='buy')`
  - `svm_gamma`: `CategoricalParameter(['scale', 0.001, 0.01, 0.1, 1.0], default='scale', space='buy')`
  - `svm_epsilon`: `DecimalParameter(0.01, 0.5, default=0.1, space='buy')`
  - **Space**: `'buy'` (valid hyperopt space; `'model'` doesn't exist in Freqtrade)
- **Logger**: Fixed missing `logger = logging.getLogger(__name__)` (was in removed `SVRSurfModel`)
- **Entry/Exit Logic**: Unchanged (Omega quality score, opposite-direction exits, multi-criteria)
- **Sizing**: Unchanged (Omega × consistency × trend, 0.3x-2.5x)

### 3. Configuration: `config_v1_27.json`
**Location**: `user_data/versions/v1.27/config/config_v1_27.json`

**FreqAI Section**:
```json
"freqai": {
  "enabled": true,
  "freqaimodel": "SVRSurfModel",
  "freqaimodel_path": "user_data/versions/v1.27/freqaimodels/",  // Custom path
  "identifier": "surf_v1_27_svm_rbf",
  "train_period_days": 7,
  "backtest_period_days": 1,
  "feature_parameters": {
    "principal_component_analysis": true,  // ENABLED (reduces dimensionality for RBF)
    "use_SVM_to_remove_outliers": true,
    "include_timeframes": ["1m", "5m", "15m"],
    ...
  },
  "model_training_parameters": {
    "C": 1.0,        // Hyperopt tunes these
    "gamma": "scale",
    "epsilon": 0.1
  }
}
```

**Key Changes**:
- `"freqaimodel_path"`: Points to v1.27-specific `freqaimodels/` (version isolation)
- `"principal_component_analysis": true`: Reduces feature space (e.g., ~50 features → 10-20 components), improves RBF efficiency
- `"model_training_parameters"`: RBF-specific defaults (hyperopt overrides from strategy params)

## Data Pipeline

### Features (from v1.26 strategy)
**Populated in `populate_indicators()`**:
1. **Multi-Scale Omega** (12, 60, 240 bars):
   - `omega_12_short`, `omega_12_long` (raw + log)
   - `omega_60_short`, `omega_60_long`
   - `omega_240_short`, `omega_240_long`
2. **Mandelbrot Metrics**:
   - `scale_consistency`: Omega correlation across scales (fractal self-similarity)
   - `trend_strength`: Directional bias (long/short Omega difference)
3. **Technical**:
   - Spanning extrema (peaks/troughs) via `identify_spanning_extrema()`
   - SURF acceleration via `calculate_surf_acceleration()`
   - Chop regime via `classify_chop_regime()`
   - RSI, Bollinger Bands, ATR, EMA
4. **FreqAI Auto-Features**:
   - Shifted candles (±2 bars)
   - Correlation pairs (BTC/USDC:USDC)
   - Multiple timeframes (1m, 5m, 15m)

**Total**: ~50-80 features (before PCA)

**After PCA**: ~10-20 principal components (captures 95% variance, improves RBF training speed)

### Target Label
**`&-omega_return_pred`**: Forward 12-bar Omega ratio (gain/loss sums, threshold 0)
- **Calculation**: `populate_any_indicators()` computes 12-bar forward returns, then Omega
- **Range**: Typically [-0.5, 1.5] (clipped to [-1, 1] for stability)
- **Why Omega?**: Captures fat tails, asymmetry (not Sharpe/Sortino which assume Gaussian)

### SVR RBF Model
**Training** (per pair, every `live_retrain_hours=1` or backtest period):
1. FreqAI splits data: 75% train, 25% test (`test_size=0.25`)
2. PCA: Reduces features to 10-20 components (if `principal_component_analysis: true`)
3. `SVRSurfModel.fit()`:
   - Scales PCA components via `StandardScaler` (mean=0, std=1)
   - Fits `SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)` on scaled train data
   - RBF kernel: `K(x, x') = exp(-gamma * ||x - x'||^2)` (distance-based similarity)
4. Saves model to `user_data/models/surf_v1_27_svm_rbf/{pair}/`

**Prediction** (every candle):
1. `SVRSurfModel.predict()`:
   - Loads model for pair
   - Scales current features
   - Predicts Omega: `pred = model.predict(X)` (continuous value)
2. FreqAI: Filters outliers via Dissimilarity Index (DI < 1.0)
3. Strategy: Uses `&-omega_return_pred` in entry/exit logic

## Entry Logic (from v1.26)
**Long Entry** (when all true):
1. **Omega Quality Score ≥ 7** (out of 15):
   - Multi-scale Omega consistency (≤3)
   - Positive trend strength (≥0.1)
   - Short-term Omega (12-bar) > threshold (0.2-0.6 per pair)
   - Near spanning trough (peak_dist ≤1)
   - RSI oversold (≤35)
   - Below Bollinger lower band
   - Acceleration positive
2. **FreqAI**: `&-omega_return_pred > 0.1` (predicted positive Omega)
3. **DI**: `do_predict == 1` (not an outlier)

**Short Entry**: Symmetric (negative Omega, peak, overbought)

## Exit Logic (Quality-Mirrored, from v1.26)
**Long Exit** (any true):
1. **Opposite Quality Score High** (short_quality_score ≥ 0.7 × entry_threshold):
   - Perfect symmetry: Exit long at peak when short conditions strong
2. **Omega Flip** (predicted Omega < -0.05)
3. **Multi-Criteria**: Acceleration negative + complexity high
4. **ROI/Stoploss**: Aggressive (2.0% immediate, 0.5% at 90min, -2.0% hard stop)

**Short Exit**: Symmetric (long quality score high, Omega positive, etc.)

## Position Sizing (Multi-Factor, from v1.26)
```python
size_multiplier = (
    omega_factor * (0.5 + abs(omega_pred))  # 0.5-1.5x
    * consistency_factor * (1.2 - scale_consistency/2)  # 0.7-1.2x
    * trend_factor * (0.5 + abs(trend_strength))  # 0.5-1.5x
)
# Result: 0.3x - 2.5x base stake
```
- High predicted Omega + consistent scales + strong trend = larger position
- Conservative: Caps at 2.5x (prevents over-leverage on extreme signals)

## Hyperoptimization

### Parameters Tuned
**Buy Space** (entry thresholds + RBF params):
- `omega_threshold_btc`, `omega_threshold_eth`, etc. (0.2-0.6)
- `entry_quality_threshold` (5-10)
- `svm_C` (0.1-10.0)
- `svm_gamma` (['scale', 0.001, 0.01, 0.1, 1.0])
- `svm_epsilon` (0.01-0.5)

**Sell Space** (exit thresholds):
- `exit_quality_threshold` (3-8)
- `exit_omega_flip_threshold` (-0.2 to -0.01)

**ROI Space** (time-based profits):
- `roi_p1`, `roi_p2`, `roi_p3`, `roi_t1`, `roi_t2`, `roi_t3`

**Stoploss Space**:
- `stoploss` (-0.05 to -0.01)

### Command
```bash
cd /Users/williamgleim/Development/07.08.25/dfai-freqtrade/freqtrade
source .venv/bin/activate
python3 -m freqtrade hyperopt \
  --config user_data/versions/v1.27/config/config_v1_27.json \
  --hyperopt-loss OnlyProfitHyperOptLoss \
  --strategy SurfMultiModel_v1_27 \
  --spaces buy sell roi stoploss \
  --epochs 200 \
  --timerange=20250709-20250930 \
  --hyperopt-database user_data/versions/v1.27/hyperopt_v1.27.pickle
```
- **Timerange**: 20250709-20250930 (~90 days, matches downloaded data)
- **Epochs**: 200 (tunes RBF + entry/exit over 200 param combinations)
- **Loss**: `OnlyProfitHyperOptLoss` (maximizes total profit)
- **Database**: `.pickle` for resume (if interrupted, re-run same command)

### Expected Output
```
Epoch 1/200: {'svm_C': 1.0, 'svm_gamma': 'scale', ...} → Total profit: 12.3%
Epoch 2/200: {'svm_C': 2.5, 'svm_gamma': 0.01, ...} → Total profit: 15.7%
...
Best epoch: 157 → svm_C=3.2, svm_gamma=0.1, epsilon=0.05, profit=18.9%
```
- **Time**: 1-3 hours (single-thread; use `--hyperopt-jobs 4` + `pip install ray` for parallel)
- **View**: `python3 -m freqtrade hyperopt-show --config ... --best 5`
- **Update Config**: Copy best params to `model_training_parameters` (C, gamma, epsilon)

## Backtesting

### Command (After Hyperopt)
```bash
cd /Users/williamgleim/Development/07.08.25/dfai-freqtrade/freqtrade
source .venv/bin/activate
python3 -m freqtrade backtesting \
  --config user_data/versions/v1.27/config/config_v1_27.json \
  --strategy SurfMultiModel_v1_27 \
  --timerange=20250930-20251007 \
  --export trades \
  --export-filename user_data/versions/v1.27/backtest_v1.27_svm_rbf.json
```
- **Timerange**: Out-of-sample (OOS) validation on recent data (not in hyperopt)
- **Export**: Saves trades to JSON (includes `&-omega_return_pred` for analysis)

### Expected Metrics (Target vs. v1.26)
| Metric | v1.26 Baseline | v1.27 Target | Improvement |
|--------|---------------|-------------|-------------|
| **Win Rate** | 55-58% | 58-62% | +3-4% (RBF captures non-linear Omega patterns) |
| **Avg Win/Loss** | 2.5x | 2.8-3.2x | +0.3-0.7x (quality-mirrored exits + RBF timing) |
| **Daily Profit** | 0.35-0.45% | 0.45-0.60% | +0.10-0.15% (better Omega prediction) |
| **Sharpe** | 1.8-2.2 | 2.2-2.8 | +0.4-0.6 (consistent via RBF smoothing) |
| **Max DD** | -3.5% | -2.5% | -1.0% (PCA + RBF reduce overfitting) |

### Validation
1. **Compare Predicted vs. Actual Omega**:
   - Export trades: Check `&-omega_return_pred` column
   - Scatter plot: `pred` (x) vs. `actual_omega` (y)
   - **Target**: R² > 0.6 (strong correlation), MAE < 0.15
2. **Feature Importance** (if using PCA):
   - FreqAI logs: "PCA explained variance: [0.35, 0.18, 0.12, ...]"
   - First 3 components should capture >70% variance
3. **Hyperopt Convergence**:
   - Check epochs 150-200: If profit plateaus, params converged
   - If rising, increase `--epochs 500`

## Technical Details

### Why IFreqaiModel (not BaseRegressionModel)?
- **Version Agnostic**: `IFreqaiModel` is stable across Freqtrade versions (abstract base)
- **BaseRegressionModel**: Subclass with boilerplate, but caused import issues in dev (`'No module named freqtrade.freqai.prediction_models.BaseRegressionModel'`)
- **IFreqaiModel**: Direct inheritance, explicit implementation of `train()`, `fit()`, `predict()` (more control)

### Why Separate Model File?
- **FreqAI Resolver**: Searches `user_data/freqaimodels/` (or custom `freqaimodel_path`) for `<ModelName>.py`
- **Error if Inline**: "Impossible to load FreqaiModel 'SVRSurfModel'" (resolver doesn't find class in strategy file)
- **Standard Practice**: All custom models separate (e.g., `LightGBMRegressorMultiTarget.py`, `CatboostClassifier.py`)

### PCA Impact
- **Before PCA**: 50-80 features (multi-scale Omega, shifted, correlations, TAs)
- **After PCA**: 10-20 components (95% variance)
- **Benefits**:
  - **Speed**: RBF trains faster (fewer dimensions, O(n²) to O(m²), m << n)
  - **Generalization**: Reduces noise (overfitting on irrelevant features)
  - **Interpretability**: Top components often = "Omega momentum", "Multi-TF trend", "Volatility regime"
- **Tradeoff**: Loses exact feature names (but Omega signal preserved in PC1-PC3)

### RBF Kernel Math
**Distance-Based Similarity**:
```
K(x, x') = exp(-gamma * ||x - x'||²)
```
- `gamma = 'scale'`: `1 / (n_features * X.var())` (auto-scales to data variance)
- High gamma: Tight fit (captures local non-linearity, risk of overfit)
- Low gamma: Smooth fit (generalizes, may underfit)
- **Hyperopt**: Tests 'scale', 0.001, 0.01, 0.1, 1.0 for optimal balance

**Epsilon-Tube SVR**:
- Predictions within `±epsilon` of actual = no penalty (sparse solution)
- Larger epsilon: Smoother (fewer support vectors, faster)
- Smaller epsilon: Tighter fit (more SVs, slower, risk of noise)
- **Hyperopt**: 0.01-0.5 (default 0.1 = balance)

**C (Regularization)**:
- High C: Fit training data closely (risk of overfit)
- Low C: Wider margin (generalizes, may underfit)
- **Hyperopt**: 0.1-10.0 (default 1.0 = standard)

## Troubleshooting

### Import Errors
**"No module named 'freqtrade.freqai.prediction_models.BaseRegressionModel'"**:
- **Cause**: Global binary vs. local dev clone conflict
- **Fix**: Always use `python3 -m freqtrade` (module mode) + venv activation
- **Alternative**: `pip install -e .` (editable install from Freqtrade root)

**"Impossible to load FreqaiModel 'SVRSurfModel'"**:
- **Cause**: Model not in `freqaimodels/` or missing `freqaimodel_path`
- **Fix**: Check `ls user_data/versions/v1.27/freqaimodels/SVRSurfModel.py` exists
- **Config**: Verify `"freqaimodel_path": "user_data/versions/v1.27/freqaimodels/"`

### Data Issues
**"No history for ... found"**:
- **Cause**: Timerange mismatch (2025 data vs. 2024 request) or missing download
- **Fix**: Use `--days 90` (last 90 days) or `--erase` to re-download
- **Check**: `python3 -m freqtrade list-data --config ...` (shows available dates)

**Rate Limit 429 Errors**:
- **Expected**: Hyperliquid throttles aggressive downloads (30 req/min limit)
- **Freqtrade**: Retries automatically (logs "Retrying still for X times")
- **Rate-Limit Script**: `download_data_incremental.py` (3s delay, 2-day chunks)

### Hyperopt
**"error: argument --spaces: invalid choice: 'model'"**:
- **Cause**: `'model'` is not a valid hyperopt space (valid: buy, sell, roi, stoploss, protection)
- **Fix**: Changed `space='model'` to `space='buy'` for `svm_C`, `svm_gamma`, `svm_epsilon`

**Epochs Slow (1-3 hours)**:
- **Normal**: FreqAI trains models per pair per epoch (BTC, ETH, HYPE, SOL = 4 models × 200 epochs = 800 trains)
- **Speed Up**: `--hyperopt-jobs 4` (parallel, requires `pip install ray`)

## Next Steps

### 1. Monitor Download
```bash
cd /Users/williamgleim/Development/07.08.25/dfai-freqtrade/freqtrade
tail -f download_erase_v1_27.log
```
- Expected: "Downloaded data for <pair> with length ~5000" (per TF)
- Wait for all pairs (BTC, ETH, HYPE, SOL) × TFs (1m, 5m, 15m) + funding/mark
- **Done**: When log shows "Downloaded data for SOL/USDC:USDC ... mark"

### 2. Run Hyperopt
```bash
python3 -m freqtrade hyperopt --config user_data/versions/v1.27/config/config_v1_27.json --hyperopt-loss OnlyProfitHyperOptLoss --strategy SurfMultiModel_v1_27 --spaces buy sell roi stoploss --epochs 200 --timerange=20250709-20250930 --hyperopt-database user_data/versions/v1.27/hyperopt_v1.27.pickle
```
- Expected: Epochs progress, logs "SVR RBF trained for <pair>: C=X, gamma=Y"
- **View Best**: `python3 -m freqtrade hyperopt-show --config ... --best 1`

### 3. Update Config with Best Params
Edit `config_v1_27.json`:
```json
"model_training_parameters": {
  "C": 3.2,      // From hyperopt best
  "gamma": 0.1,
  "epsilon": 0.05
}
```

### 4. Backtest OOS
```bash
python3 -m freqtrade backtesting --config user_data/versions/v1.27/config/config_v1_27.json --strategy SurfMultiModel_v1_27 --timerange=20250930-20251007 --export trades --export-filename user_data/versions/v1.27/backtest_v1.27_svm_rbf.json
```
- Expected: Report with trades, win rate ~58%, profit ~0.45-0.60% daily

### 5. Analyze Results
- **Prediction Accuracy**: Compare `&-omega_return_pred` vs. actual in export JSON
- **Feature Importance**: Check FreqAI logs for PCA variance (top 3 components)
- **Trades**: Review entry/exit quality scores (logs show per-trade reasons)

### 6. Deploy (If Successful)
- Stop v1.26: `pkill -f "freqtrade trade.*v1.26"`
- Run v1.27: `python3 -m freqtrade trade --config user_data/versions/v1.27/config/config_v1_27.json --strategy SurfMultiModel_v1_27`
- Monitor: Logs (`tail -f user_data/versions/v1.27/logs/freqtrade_v1.27.log`), API (`http://localhost:8080`)

## References
- **FreqAI Docs**: https://www.freqtrade.io/en/stable/freqai/
- **SVR Sklearn**: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
- **RBF Kernel**: https://scikit-learn.org/stable/modules/svm.html#rbf-kernel
- **PCA**: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- **Omega Ratio**: Keating & Shadwick (2002), "A Universal Performance Measure"

## Changelog
- **2025-10-07 22:00**: Created v1.27 structure (strategy, config, model file)
- **2025-10-07 22:15**: Fixed logger import in strategy, added hyperopt params
- **2025-10-07 22:20**: Moved SVRSurfModel to separate file (`freqaimodels/SVRSurfModel.py`)
- **2025-10-07 22:30**: Started data download (90 days Hyperliquid futures)
- **2025-10-07 22:35**: Download progressing (rate-limited, ~60 min ETA)
- **Next**: Hyperopt → backtest → deploy

---
**Author**: Strategy Evolution Framework  
**Version**: 1.27  
**Date**: October 7, 2025  
**Status**: Development - Data Download in Progress

