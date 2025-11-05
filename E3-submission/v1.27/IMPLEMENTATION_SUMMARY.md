# V1.27 Implementation Summary: SVM + RBF for FreqAI

## What Was Built
Integrated a Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel into FreqAI for predicting Omega ratio values in v1.27. This is a minimal enhancement to v1.26's quality-mirrored exit strategy.

## Files Created/Modified

### 1. Custom FreqAI Model
**File**: `user_data/versions/v1.27/freqaimodels/SVRSurfModel.py`
- **Class**: `SVRSurfModel(IFreqaiModel)`
- **Pipeline**: `StandardScaler` → `SVR(kernel='rbf', C, gamma, epsilon)`
- **Methods**: `train()`, `fit()`, `predict()` (implements IFreqaiModel interface)
- **Parameters**: Reads from config's `model_training_parameters` (C, gamma, epsilon)
- **Logging**: Logs training params per pair

### 2. Strategy
**File**: `user_data/versions/v1.27/strategy/SurfMultiModel_v1_27.py`
- **Base**: Copied from v1.26 (quality-mirrored exits, multi-factor sizing)
- **Changes**:
  - Removed inline model class (moved to separate file)
  - Added logger: `logger = logging.getLogger(__name__)`
  - Added hyperopt params for RBF:
    - `svm_C`: DecimalParameter(0.1, 10.0, default=1.0, space='buy')
    - `svm_gamma`: CategoricalParameter(['scale', 0.001, 0.01, 0.1, 1.0], default='scale', space='buy')
    - `svm_epsilon`: DecimalParameter(0.01, 0.5, default=0.1, space='buy')
  - No changes to entry/exit logic, sizing, or Omega calculations

### 3. Configuration
**File**: `user_data/versions/v1.27/config/config_v1_27.json`
- **FreqAI**:
  - `"freqaimodel": "SVRSurfModel"`
  - `"freqaimodel_path": "user_data/versions/v1.27/freqaimodels/"` (version-specific)
  - `"principal_component_analysis": true` (enabled for RBF efficiency)
  - `"model_training_parameters": {"C": 1.0, "gamma": "scale", "epsilon": 0.1}` (RBF defaults)
- **Data**: `"datadir": "user_data/data"` (shared across versions)
- **DB**: `"db_url": "sqlite:///tradesv3_v1.27.dryrun.sqlite"` (isolated)

### 4. Documentation
**Files**:
- `DESIGN_v1_27_SVM_RBF.md`: Complete design doc (architecture, pipeline, hyperopt, troubleshooting)
- `IMPLEMENTATION_SUMMARY.md`: This file (quick reference)

## Key Technical Decisions

### 1. IFreqaiModel (not BaseRegressionModel)
- **Reason**: Import issues with `BaseRegressionModel` in dev environment (`'No module named freqtrade.freqai.prediction_models.BaseRegressionModel'`)
- **Solution**: Inherit directly from `IFreqaiModel` (abstract base, stable across versions)
- **Trade-off**: More boilerplate (explicit `train()`, `fit()`, `predict()`), but robust

### 2. Separate Model File
- **Reason**: FreqAI's model resolver searches `freqaimodels/` directory (not strategy files)
- **Error**: "Impossible to load FreqaiModel 'SVRSurfModel'" when model was inline
- **Solution**: Moved to `user_data/versions/v1.27/freqaimodels/SVRSurfModel.py`
- **Standard**: All custom FreqAI models use this structure

### 3. PCA Enabled
- **Reason**: RBF kernels scale O(n²) with features (50-80 features = slow)
- **Benefit**: PCA reduces to 10-20 components (95% variance), faster training
- **Trade-off**: Loses exact feature names, but Omega signal preserved in top components

### 4. Hyperopt Space 'buy' (not 'model')
- **Reason**: `'model'` is not a valid hyperopt space in Freqtrade (valid: buy, sell, roi, stoploss, protection)
- **Error**: "argument --spaces: invalid choice: 'model'"
- **Solution**: Changed `space='model'` to `space='buy'` for RBF params

### 5. StandardScaler in Pipeline
- **Reason**: RBF is distance-based (sensitive to feature scales)
- **Without Scaling**: Features with large ranges (e.g., price) dominate kernel
- **With Scaling**: All features normalized (mean=0, std=1), equal weighting

## Implementation Steps Completed

### ✅ Step 1: Create v1.27 Structure
```bash
mkdir -p user_data/versions/v1.27/{strategy,config,logs,freqaimodels}
```

### ✅ Step 2: Copy & Adapt Strategy
- Copied `SurfMultiModel_v1_26.py` → `SurfMultiModel_v1_27.py`
- Added RBF hyperopt params (C, gamma, epsilon)
- Fixed logger import
- Removed inline model class

### ✅ Step 3: Create Custom Model
- Created `SVRSurfModel.py` in `freqaimodels/`
- Implemented `IFreqaiModel` interface (train, fit, predict)
- Built pipeline: StandardScaler → SVR(kernel='rbf')
- Read params from config's `model_training_parameters`

### ✅ Step 4: Update Configuration
- Set `"freqaimodel": "SVRSurfModel"`
- Added `"freqaimodel_path"` (version-specific)
- Enabled PCA: `"principal_component_analysis": true`
- Set RBF defaults in `model_training_parameters`

### ✅ Step 5: Download Data
```bash
python3 -m freqtrade download-data \
  --config user_data/versions/v1.27/config/config_v1_27.json \
  --erase --days 90 --trading-mode futures --exchange hyperliquid \
  --pairs BTC/USDC:USDC ETH/USDC:USDC HYPE/USDC:USDC SOL/USDC:USDC \
  --timeframes 1m 5m 15m --data-format-ohlcv feather
```
- **Status**: In progress (~60 min ETA, rate-limited)
- **Log**: `download_erase_v1_27.log`

### ⏳ Step 6: Hyperopt (Pending Data Download)
```bash
python3 -m freqtrade hyperopt \
  --config user_data/versions/v1.27/config/config_v1_27.json \
  --hyperopt-loss OnlyProfitHyperOptLoss \
  --strategy SurfMultiModel_v1_27 \
  --spaces buy sell roi stoploss \
  --epochs 200 \
  --timerange=20250709-20250930 \
  --hyperopt-database user_data/versions/v1.27/hyperopt_v1.27.pickle
```
- **Expected**: Tunes RBF params (C, gamma, epsilon) + entry/exit thresholds
- **Time**: 1-3 hours (single-thread)
- **Output**: Best params to copy to config

### ⏳ Step 7: Backtest (Pending Hyperopt)
```bash
python3 -m freqtrade backtesting \
  --config user_data/versions/v1.27/config/config_v1_27.json \
  --strategy SurfMultiModel_v1_27 \
  --timerange=20250930-20251007 \
  --export trades \
  --export-filename user_data/versions/v1.27/backtest_v1.27_svm_rbf.json
```
- **Expected**: OOS validation, win rate ~58-62%, profit ~0.45-0.60% daily

## Troubleshooting Issues Resolved

### 1. Import Errors
**Error**: `"No module named 'freqtrade.freqai.prediction_models.BaseRegressionModel'"`
- **Root Cause**: Global binary vs. local dev clone conflict
- **Fix**: Always use `python3 -m freqtrade` (module mode) + venv activation
- **Alternative**: Switched to `IFreqaiModel` (more robust)

### 2. Model Loading Error
**Error**: `"Impossible to load FreqaiModel 'SVRSurfModel'"`
- **Root Cause**: Model class was in strategy file (FreqAI resolver doesn't search there)
- **Fix**: Moved to `user_data/versions/v1.27/freqaimodels/SVRSurfModel.py`

### 3. Logger NameError
**Error**: `"NameError: name 'logger' is not defined"`
- **Root Cause**: Logger was in removed `SVRSurfModel` class
- **Fix**: Added `logger = logging.getLogger(__name__)` to strategy file

### 4. Missing Imports
**Errors**: `"name 'Any' is not defined"`, `"name 'CategoricalParameter' is not defined"`
- **Fix**: Added `from typing import Any`, `from freqtrade.strategy import ..., CategoricalParameter`

### 5. Hyperopt Space Error
**Error**: `"argument --spaces: invalid choice: 'model'"`
- **Root Cause**: `'model'` is not a valid hyperopt space
- **Fix**: Changed `space='model'` to `space='buy'` for RBF params

### 6. Data Availability Issues
**Error**: `"No history for ... found"` (2025 timerange)
- **Root Cause**: Hyperliquid doesn't have data for early 2025 (requested 20250701-20250901)
- **Fix**: Used `--days 90` (last 90 days = 20250709-20251007, real data)

## Testing Plan

### 1. Strategy Load Test (Dry-Run)
```bash
python3 -m freqtrade backtesting --config user_data/versions/v1.27/config/config_v1_27.json --strategy SurfMultiModel_v1_27 --timerange 20250709-20250710 --dry-run --dry-run-wallet 250000
```
- **Expected**: Strategy + model load, no errors
- **Status**: Pending data download completion

### 2. Hyperopt Convergence
- **Check**: Epochs 150-200 should plateau (if rising, increase `--epochs 500`)
- **View**: `python3 -m freqtrade hyperopt-show --config ... --best 5`

### 3. Prediction Accuracy (Backtest)
- **Export**: JSON with `&-omega_return_pred` column
- **Scatter Plot**: Predicted vs. actual Omega
- **Target**: R² > 0.6, MAE < 0.15

### 4. Feature Importance (PCA)
- **Logs**: "PCA explained variance: [0.35, 0.18, 0.12, ...]"
- **Target**: Top 3 components capture >70% variance

## Expected Performance

| Metric | v1.26 Baseline | v1.27 Target | Improvement |
|--------|---------------|-------------|-------------|
| Win Rate | 55-58% | 58-62% | +3-4% |
| Avg Win/Loss | 2.5x | 2.8-3.2x | +0.3-0.7x |
| Daily Profit | 0.35-0.45% | 0.45-0.60% | +0.10-0.15% |
| Sharpe | 1.8-2.2 | 2.2-2.8 | +0.4-0.6 |
| Max DD | -3.5% | -2.5% | -1.0% |

**Why Improvement?**:
- RBF captures non-linear Omega patterns (multi-scale interactions)
- PCA reduces overfitting (noise reduction)
- Quality-mirrored exits + RBF timing = higher win/loss ratio

## Next Actions

### Immediate (After Data Download)
1. **Check Download Completion**:
   ```bash
   tail -f download_erase_v1_27.log  # Wait for "Downloaded data for SOL/USDC:USDC ... mark"
   ```

2. **Verify Data**:
   ```bash
   python3 -m freqtrade list-data --config user_data/versions/v1.27/config/config_v1_27.json
   ```
   - Expected: BTC, ETH, HYPE, SOL × 1m, 5m, 15m (~5000 candles each)

3. **Run Hyperopt**:
   ```bash
   python3 -m freqtrade hyperopt --config ... --epochs 200 --timerange=20250709-20250930
   ```

### After Hyperopt
4. **Update Config**: Copy best RBF params to `model_training_parameters`

5. **Backtest OOS**: Validate on 20250930-20251007

6. **Analyze**: Check prediction accuracy, PCA variance, trade quality

### Deployment (If Successful)
7. **Stop v1.26**: `pkill -f "freqtrade trade.*v1.26"`

8. **Run v1.27**:
   ```bash
   python3 -m freqtrade trade --config user_data/versions/v1.27/config/config_v1_27.json --strategy SurfMultiModel_v1_27
   ```

9. **Monitor**: Logs (`tail -f user_data/versions/v1.27/logs/freqtrade_v1.27.log`), API (`http://localhost:8080`)

## References
- Design Doc: `DESIGN_v1_27_SVM_RBF.md` (full architecture, math, troubleshooting)
- Strategy: `strategy/SurfMultiModel_v1_27.py` (entry/exit logic, hyperopt params)
- Model: `freqaimodels/SVRSurfModel.py` (RBF pipeline, IFreqaiModel implementation)
- Config: `config/config_v1_27.json` (FreqAI settings, RBF defaults)

---
**Status**: Data download in progress (~60 min ETA) → Hyperopt → Backtest → Deploy  
**Date**: October 7, 2025  
**Version**: 1.27

