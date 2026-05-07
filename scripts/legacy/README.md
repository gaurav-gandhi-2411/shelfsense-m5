# Legacy Scripts

These scripts ran the Phase 1–2 experiments. They are kept for reference but no longer maintained. Use the `shelfsense` CLI for current work.

| Legacy script | New CLI invocation |
|---|---|
| `05_build_features.py` | `shelfsense features build` |
| `06_train_lightgbm.py` | `shelfsense train tweedie-mh` (Day 6 global baseline — superseded by multi-horizon) |
| `07_train_per_category.py` | (deprecated — superseded by `shelfsense train store-dept`) |
| `08_recursive_v2.py` | `shelfsense submit --variant recursive` (Stage 4) |
| `09_train_multi_horizon.py` | `shelfsense train tweedie-mh` |
| `10_train_per_store.py` | `shelfsense train per-store` |
| `11_lgbm_per_dept.py` | `shelfsense train per-dept` |
| `12_lgbm_tvp_sweep.py` | `shelfsense train tweedie-mh --tvp <value>` |
| `13_ensemble_tvp.py` | `shelfsense ensemble` |
| `14_lgbm_rmse_mh.py` | `shelfsense train rmse-mh` (Stage 4 — not yet wired) |
| `15_sanity_train_lags.py` | (one-off diagnostic — not retained) |
| `16_lgbm_tvp_multiseed.py` | (deprecated — abandoned in Phase 2) |
| `17_lgbm_store_dept.py` | `shelfsense train store-dept` |
| `18_lgbm_ylags_mh.py` | `shelfsense train tweedie-mh +features=ylags` |
| `_bench_slice.py` | (one-off benchmark — not retained) |
| `run_day3.py` | (Day 3 classical stats — no CLI equivalent planned) |
| `run_day4.py` | (Day 4 Prophet — no CLI equivalent planned) |
| `smoke_test_nbeats.py` | (DL workstream — Stage 4 if pursued) |
| `generate_charts.py` | `shelfsense report --regenerate-charts` (Stage 6) |
