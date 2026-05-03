# Not Submitted — WS2.5 Tweedie Ensemble Candidates

These 4 ensemble CSVs were built by `scripts/13_ensemble_tvp.py` (commit d269f04)
but intentionally not submitted to Kaggle.

## Files

- `ens_tvp13_mhblend_5050.csv` — 0.5 × tvp=1.3 + 0.5 × mh_blend eval preds
- `ens_tvp13_tvp17_5050.csv`   — 0.5 × tvp=1.3 + 0.5 × tvp=1.7 eval preds
- `ens_tvp13_tvp17_mh_3333.csv`— 1/3 × each of tvp=1.3, tvp=1.7, mh_blend
- `ens_optuna_blend.csv`       — Optuna 50-trial weights: tvp=1.3=0.913, mh=0.075, tvp=1.7=0.012

## Why not submitted

Initial run used oracle bias: mh_blend val rows were extracted from the
submission CSV (single-step lgbm_best.pkl predictions with actual per-day
features), while tvp variants used multi-horizon from d_1913. This made
Optuna collapse to 0.966 mh_blend weight, which was meaningless.

After the same-origin fix (reload Day 9 pkl models, predict from d_1913),
all 3 variants scored on a comparable basis:
  tvp=1.3:  0.6860
  mh_blend: 0.7210
  tvp=1.7:  0.7713

Every ensemble candidate scored worse than pure tvp=1.3 on val:
  ens_tvp13_mhblend_5050: 0.7001
  ens_optuna_blend:        0.6883  (best, still > 0.6860 pure tvp=1.3)

The three Tweedie variants share nearly identical feature structure and
training data — the only difference is tvp. Without complementary error
structure, blending provides no diversity benefit on val. Private LB for
tvp=1.3 is 0.5693 (current best); no ensemble would improve on this.
