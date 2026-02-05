# FIX LOG - Atari Interference Ensemble Issue

**Date:** February 5, 2026
**Issue:** Interference Ensemble failed on Atari environments

## Problem Description

```
RuntimeError: The size of tensor a (20) must match the size of tensor b (5)
at non-singleton dimension 2
```

## Root Cause

In `src/quantum_inspired/interference_ensemble.py`, line 326:

```python
# BEFORE (broken):
weights_expanded = weights.view(self.num_models, 1, 1, 1)
```

This hard-coded 4 dimensions, assuming predictions had shape:
- `(num_models, batch, seq, obs_dim)`

For Atari with CNN outputs, predictions have shape:
- `(num_models, batch, seq, H, W)` or
- `(num_models, batch, seq, channels, H, W)`

The mismatch caused the tensor broadcast to fail.

## Fix Applied

```python
# AFTER (fixed):
num_extra_dims = predictions.dim() - 1  # All dims except num_models
weights_shape = [self.num_models] + [1] * num_extra_dims
weights_expanded = weights.view(*weights_shape)
```

This dynamically determines the number of dimensions and creates the appropriate shape.

## Files Modified

- `src/quantum_inspired/interference_ensemble.py` (line 326-330)

## Verification

The fix:
1. Works for state-based obs: `(5, 32, 20, 24)` → weights `(5, 1, 1, 1)`
2. Works for visual obs: `(5, 16, 20, 84, 84)` → weights `(5, 1, 1, 1, 1)`
3. Works for any dimension count

## Impact

- Atari experiments (Pong, Breakout) should now work with Interference Ensemble
- No impact on DMControl experiments (already working)
- No impact on CartPole experiments (already working)

## Recommendation

Re-run Phase 3 Atari experiments with the fixed code to obtain complete results.

---

*Fix documented: February 5, 2026*
