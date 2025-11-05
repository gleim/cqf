# Tier2 Moderate Loss Optimization - Implementation Complete

**Date**: October 7, 2025  
**Status**: Implemented in v1.27  
**Change Type**: Exit logic enhancement (simple, low-risk)

## What Changed

Added **scale consistency check** to tier2_moderate_loss exit logic.

### Before (2 conditions):
```python
if current_omega < -0.20 and surf_accel < -0.012:
    return "tier2_moderate_loss"
```

### After (3 conditions):
```python
if (current_omega < -0.20 and 
    surf_accel < -0.012 and
    current_consistency < 0.25):  # NEW
    return "tier2_moderate_loss"
```

## Rationale

**Problem**: Tier2 was exiting on Omega + Acceleration collapse, but ignoring **scale consistency** (fractal complexity). This caused premature exits during temporary chop where recovery was likely.

**Solution**: Only exit tier2 if consistency is also degraded (<0.25), indicating genuine quality collapse (not just noise).

**Logic**:
- **High Consistency (≥0.25)**: Fractal structure intact → Omega dip temporary → Hold
- **Low Consistency (<0.25)**: Market structure breaking → Omega collapse permanent → Exit

## Expected Impact

### Metrics (Tier2-Specific):
- **Tier2 Exit Count**: ↓ 10-20% (fewer false exits in choppy markets)
- **Avg Loss per Tier2**: Slightly worse (-0.75% → -0.80%) but offset by recoveries
- **Overall Win Rate**: ↑ 0.5-1.5% (recovered trades now profitable)
- **Sharpe Ratio**: Slight increase (fewer premature exits = smoother equity)

### Trade-offs:
- **Slightly More Losses**: May hold 0.1-0.2% longer if consistency stays high but Omega doesn't recover
- **Increased Exposure**: Fewer exits = slightly longer avg trade duration
- **Risk**: Low (conservative change, aligns with tier3 which already uses 3 conditions)

## Implementation Details

### File Modified:
`user_data/versions/v1.27/strategy/SurfMultiModel_v1_27.py`

### Lines Changed:
Lines 1141-1158 (tier2 custom_exit logic)

### Code Changes:
1. Added `current_consistency < 0.25` condition to both long and short tier2 exits
2. Updated logger to show consistency value in exit message
3. Updated comment to reflect "ALL THREE" conditions (was "BOTH")

### Consistency Threshold:
- **Fixed at 0.25** (middle ground between tier2 "moderate" and tier3 "extreme")
- **Not hyperopt-tunable** (keeping it simple as requested)
- **Can be made tunable** if backtests show need for optimization (add `tier2_consistency_threshold` parameter)

## Testing Plan

### 1. Quick Validation (Dry-Run)
```bash
cd /Users/williamgleim/Development/07.08.25/dfai-freqtrade/freqtrade
source .venv/bin/activate

# Test strategy loads without errors
python3 -m freqtrade backtesting \
  --config user_data/versions/v1.27/config/config_v1_27.json \
  --strategy SurfMultiModel_v1_27 \
  --timerange 20251005-20251007 \
  --dry-run --dry-run-wallet 250000

# Expected: No errors, strategy loads, tier2 logic validates
```

### 2. Backtest Comparison (Before/After) - When Data Available
```bash
# Backtest with optimization (current code)
python3 -m freqtrade backtesting \
  --config user_data/versions/v1.27/config/config_v1_27.json \
  --strategy SurfMultiModel_v1_27 \
  --freqaimodel SVRSurfModel \
  --freqaimodel-path user_data/versions/v1.27/freqaimodels/ \
  --timerange=20240601-20240831 \
  --export trades \
  --export-filename user_data/versions/v1.27/backtest_tier2_optimized.json

# Analyze tier2 exits:
grep "tier2_moderate_loss" user_data/versions/v1.27/backtest_tier2_optimized.json | wc -l  # Count
# Compare to baseline (if available): Should show 10-20% fewer tier2 exits
```

### 3. Monitor in Live (If Deployed)
Watch tier2 exit logs for consistency values:
```bash
tail -f user_data/versions/v1.27/logs/freqtrade_v1.27.log | grep "TIER 2"

# Expected logs:
# [BTC/USDC:USDC] TIER 2: Moderate loss exit (profit: -0.75%, omega: -0.22, consistency: 0.18)
# Verify consistency is < 0.25 for all tier2 exits
```

## Tier Structure Summary (Updated)

| Tier | Loss Range | Exit Criteria | Selectivity |
|------|-----------|---------------|------------|
| **Tier 0** | > +0.5% | Omega flip + Strong accel | Medium (2 conditions) |
| **Tier 1** | < -1.0% | Omega collapsed | Loose (1 condition) |
| **Tier 2** | -1.0% to -0.5% | Omega + Accel + **Consistency** | **Tighter** (3 conditions) |
| **Tier 3** | -0.5% to -0.2% | Omega + Accel + Consistency | Tightest (3 conditions) |

**Alignment**: Tier2 now mirrors tier3 structure (all 3 metrics), creating consistent logic across loss tiers.

## Rollback Instructions (If Needed)

If backtests show worse performance (unlikely), revert by removing consistency check:

```python
# Change lines 1147-1152 and 1154-1158 back to:
if current_omega < self.tier2_omega_threshold.value and surf_accel < self.tier2_accel_threshold.value:
    logger.info(f"[{pair}] TIER 2: Moderate loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f})")
    return "tier2_moderate_loss"
```

## Next Steps

1. **Wait for Data**: Need 7+ days of historical data for FreqAI (see STATUS.md)
2. **Backtest**: Run comparison when data available (Option 1 from STATUS.md)
3. **Hyperopt**: Tune all parameters including tier2 thresholds (but consistency stays at 0.25)
4. **Deploy**: If metrics improve (expected), deploy to paper trading
5. **Monitor**: Track tier2 exit frequency, recovery rate, consistency values in logs

## Alternative Enhancements (Future)

If backtests show need for further optimization:

### Option A: Make Consistency Tunable
```python
# Add to parameters (around line 189):
tier2_consistency_threshold = DecimalParameter(0.15, 0.35, default=0.25, space='sell', optimize=True)

# In custom_exit:
current_consistency < self.tier2_consistency_threshold.value
```
- **Benefit**: Hyperopt finds optimal threshold (likely 0.22-0.28)
- **Cost**: One more parameter to tune

### Option B: Add Opposite Quality Score
```python
# Check if opposite direction setup is forming:
opposite_score = dataframe['short_score'].iloc[-1] if is_long else dataframe['long_score'].iloc[-1]

if (current_omega < self.tier2_omega_threshold.value and surf_accel < self.tier2_accel_threshold.value) or \
   (opposite_score >= self.technical_score_threshold.value * 0.6):
    return "tier2_moderate_loss"
```
- **Benefit**: Catches reversals faster (exits when opposite setup appears)
- **Cost**: More aggressive (more exits, possibly false signals)

## Documentation

- **Design**: See `TIER2_OPTIMIZATION.md` for full rationale and analysis
- **Implementation**: This file (implementation complete)
- **Strategy**: `SurfMultiModel_v1_27.py` lines 1141-1158 (tier2 custom_exit)
- **Status**: See `STATUS.md` for overall v1.27 progress (waiting on data)

---
**Author**: Strategy Evolution Framework  
**Version**: 1.27 (tier2 optimization)  
**Date**: October 7, 2025  
**Status**: Implemented, ready for testing when data available

