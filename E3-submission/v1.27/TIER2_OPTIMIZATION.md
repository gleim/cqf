# Tier2 Moderate Loss Exit - Simple Optimization Opportunities

**Date**: October 7, 2025  
**Focus**: `tier2_moderate_loss` exit tag (moderate loss -1.0% to -0.5%)

## Current Logic (v1.27)

**Condition**: Profit between -1.0% and -0.5% (moderate loss)
**Exit Criteria**: **BOTH** conditions must be true:
1. **Omega collapsed**: `current_omega < -0.20` (long) or `> 0.20` (short)
2. **Momentum reversed**: `surf_accel < -0.012` (long) or `> 0.012` (short)

**Parameters**:
```python
tier2_loss_min = -0.010  # -1.0% (range: -1.2% to -0.8%)
tier2_loss_max = -0.005  # -0.5% (range: -0.7% to -0.4%)
tier2_omega_threshold = -0.20  # (range: -0.25 to -0.15)
tier2_accel_threshold = -0.012  # (range: -0.018 to -0.008)
```

## Simple Optimization: Add Scale Consistency Check

### Problem
Current tier2 exits on **Omega + Acceleration** only, but ignores **scale consistency** (fractal complexity). This can cause premature exits during:
- **Temporary chop** (high complexity, low consistency) where Omega/accel dip but recovery likely
- **False breakdowns** (low consistency = noisy market) where reversal isn't confirmed

### Solution
Add a **3rd condition**: Only exit if scale consistency is also degraded (< 0.25), indicating genuine quality collapse (not just noise).

### Implementation (Minimal Change)

**Edit `custom_exit()` in `SurfMultiModel_v1_27.py`** (lines 1145-1153):

```python
# TIER 2: MODERATE LOSS - Quality Collapse
# ========================================================================
# Exit only if BOTH Omega collapsed AND momentum reversed AND consistency low
if self.tier2_loss_min.value <= current_profit < self.tier2_loss_max.value:
    if is_long:
        if (current_omega < self.tier2_omega_threshold.value and 
            surf_accel < self.tier2_accel_threshold.value and
            current_consistency < 0.25):  # NEW: Add consistency check
            logger.info(f"[{pair}] TIER 2: Moderate loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f}, consistency: {current_consistency:.2f})")
            return "tier2_moderate_loss"
    else:
        if (current_omega > -self.tier2_omega_threshold.value and 
            surf_accel > -self.tier2_accel_threshold.value and
            current_consistency < 0.25):  # NEW: Add consistency check
            logger.info(f"[{pair}] TIER 2: Moderate loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f}, consistency: {current_consistency:.2f})")
            return "tier2_moderate_loss"
```

**Alternatively (Hyperopt-Tunable)**:

Add a new parameter and make it tunable:

```python
# In strategy parameters (around line 189):
tier2_consistency_threshold = DecimalParameter(0.15, 0.35, default=0.25, space='sell', optimize=True)

# In custom_exit logic:
if (current_omega < self.tier2_omega_threshold.value and 
    surf_accel < self.tier2_accel_threshold.value and
    current_consistency < self.tier2_consistency_threshold.value):  # Tunable threshold
```

### Expected Impact

**Benefit**:
- **Fewer False Exits**: Holds through temporary Omega/accel dips in choppy markets (high complexity, low consistency)
- **Better Recovery Rate**: Avoids exiting when Omega might recover (consistency still decent = fractal structure intact)
- **Tighter Logic**: Aligns tier2 with tier3 (which uses all 3: Omega + accel + consistency)

**Trade-offs**:
- **Slightly More Losses**: May hold 0.1-0.2% longer before exiting (if consistency stays high but Omega/accel don't recover)
- **Increased Exposure**: Fewer exits = slightly longer average trade duration

**Hyperopt Will Decide**:
- If consistency=0.25 is too tight → Hyperopt relaxes to 0.30-0.35
- If too loose → Hyperopt tightens to 0.15-0.20
- Expected optimal: ~0.22-0.28 (between tier2 "moderate" and tier3 "extreme" thresholds)

### Why This Works

**Tier Structure** (Current vs. Optimized):

| Tier | Loss Range | Current Exit Criteria | Optimized Exit Criteria | Selectivity |
|------|-----------|----------------------|------------------------|------------|
| **Tier 1** | < -1.0% | Omega collapsed | Same (no change) | Loose (1 condition) |
| **Tier 2** | -1.0% to -0.5% | Omega + Accel | Omega + Accel + **Consistency** | **Tighter** (3 conditions) |
| **Tier 3** | -0.5% to -0.2% | Omega + Accel + Consistency | Same (no change) | Tightest (3 conditions) |

**Logic**:
- **Tier 1** (deep loss): Exit fast on Omega alone (pre-empt stoploss)
- **Tier 2** (moderate loss): Exit on confirmed quality collapse (Omega + Accel + **Consistency**)
- **Tier 3** (small loss): Only exit on extreme collapse (all 3 metrics very bad)

**Consistency as Filter**:
- **High Consistency (>0.25)**: Fractal structure intact → Omega dip may be temporary → Hold
- **Low Consistency (<0.25)**: Market structure breaking → Omega collapse likely permanent → Exit

### Alternative: Use Opposite-Direction Quality Score

**Even Simpler**: Check if **opposite direction quality score is high** (mirroring v1.26's quality-mirrored exits in `populate_exit_trend`).

```python
# TIER 2: MODERATE LOSS - Quality Collapse OR Opposite Setup
if self.tier2_loss_min.value <= current_profit < self.tier2_loss_max.value:
    # Get opposite quality score
    opposite_score = dataframe['short_score'].iloc[-1] if is_long else dataframe['long_score'].iloc[-1]
    
    if is_long:
        if (current_omega < self.tier2_omega_threshold.value and surf_accel < self.tier2_accel_threshold.value) or \
           (opposite_score >= self.technical_score_threshold.value * 0.6):  # 60% of entry threshold
            logger.info(f"[{pair}] TIER 2: Moderate loss exit (profit: {current_profit:.2%}, omega: {current_omega:.3f}, opp_score: {opposite_score:.1f})")
            return "tier2_moderate_loss"
```

**Logic**: Exit tier2 if **either**:
1. Omega + Accel collapsed (current logic), **OR**
2. Opposite direction quality score is decent (≥60% of entry threshold)

**Benefit**: Catches reversals faster (exits when SHORT setup appears during LONG moderate loss).

## Recommendation

**Start with Consistency Check (Simpler)**:
- Add `current_consistency < 0.25` to tier2 logic
- Test in backtest (compare tier2 exit count, avg loss per tier2 exit)
- If backtests show improvement (fewer false exits, better recovery rate), deploy

**If Complexity Needed**:
- Add `tier2_consistency_threshold` parameter (hyperopt-tunable)
- Let hyperopt find optimal threshold (likely 0.20-0.28)

**If Opposite Score Preferred**:
- Add opposite quality score check (mirrors v1.26 symmetry)
- More aggressive (exits faster on reversal signals)

### Implementation Steps

1. **Edit Strategy** (v1.27):
   - Add consistency check to tier2 logic (lines 1145-1153)
   - Update logger to show consistency value

2. **Backtest Comparison** (Before/After):
   ```bash
   # Before (current tier2)
   python3 -m freqtrade backtesting --config ... --timerange=... --export trades --export-filename tier2_before.json
   
   # After (consistency-filtered tier2)
   python3 -m freqtrade backtesting --config ... --timerange=... --export trades --export-filename tier2_after.json
   
   # Compare: grep "tier2_moderate_loss" tier2_before.json vs tier2_after.json
   # Check: Avg profit per tier2 exit, total tier2 exits, recovery rate after skipped exits
   ```

3. **Hyperopt** (If Adding Parameter):
   - Add `tier2_consistency_threshold` to parameters (optimize=True)
   - Run hyperopt with `--spaces sell` (tunes tier2_consistency_threshold + other sell params)

4. **Monitor Live** (If Deployed):
   - Check tier2 exit logs: Are consistency values logged correctly?
   - Track recovery rate: How many trades recover after tier2 skip (consistency > 0.25)?

### Expected Backtest Improvement

**Metrics** (Tier2-Specific):
- **Tier2 Exit Count**: Decrease by 10-20% (fewer false exits)
- **Avg Loss per Tier2 Exit**: Slightly worse (-0.75% → -0.80%) but offset by recoveries
- **Overall Win Rate**: Increase by 0.5-1.5% (recovered trades now profitable)
- **Sharpe Ratio**: Slight increase (fewer premature exits = smoother equity curve)

**If No Improvement**: Remove consistency check (current tier2 logic is already optimal).

---
**Status**: Proposed optimization, ready for implementation and testing.  
**Complexity**: Low (1-line change) or Medium (add parameter + hyperopt)  
**Risk**: Low (conservative change, aligns with tier3 logic)  
**Next Step**: Implement consistency check, backtest on 2-3 months of data, compare metrics.

