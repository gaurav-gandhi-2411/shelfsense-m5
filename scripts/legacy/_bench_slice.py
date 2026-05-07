"""
Quick benchmark: LightGBM training time on HOBBIES_2, FOODS_1, FOODS_3 slices.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "C:/Users/gaura/anaconda3/Lib/site-packages")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))

import pandas as pd, numpy as np, lightgbm as lgb, time
from features.hierarchy import CAT_DTYPES

NUM_FEATURES = [
    "weekday","month","quarter","year","day_of_month","week_of_year",
    "is_weekend","is_holiday","is_snap_ca","is_snap_tx","is_snap_wi",
    "days_since_event","days_until_next_event",
    "sell_price","price_change_pct","price_relative_mean",
    "price_volatility","has_price_change",
    "lag_7","lag_14","lag_28","lag_56",
    "roll_mean_7","roll_std_7","roll_min_7","roll_max_7",
    "roll_mean_28","roll_std_28","roll_min_28","roll_max_28",
    "roll_mean_56","roll_std_56","roll_min_56","roll_max_56",
    "roll_mean_180","roll_std_180","roll_min_180","roll_max_180",
]
CAT = ["cat_id","dept_id","store_id","state_id"]
ALL_FEATURES = NUM_FEATURES + CAT

PARAMS = {
    "objective": "tweedie", "metric": "tweedie", "verbose": -1,
    "num_threads": 0, "bagging_freq": 1, "seed": 42,
    "learning_rate": 0.025, "num_leaves": 64, "min_data_in_leaf": 50,
    "feature_fraction": 0.7, "bagging_fraction": 0.9,
    "lambda_l2": 0.1, "tweedie_variance_power": 1.3,
}

FEAT_DIR = os.path.join(PROJ, "data", "processed", "features")

for dept, store in [("HOBBIES_2","CA_1"), ("FOODS_1","CA_1"), ("FOODS_3","CA_1")]:
    cols = list(dict.fromkeys(["id","d_num","sales"] + NUM_FEATURES + CAT))
    df = pd.read_parquet(
        os.path.join(FEAT_DIR, f"store_{store}.parquet"),
        columns=cols,
        filters=[("dept_id","==",dept), ("d_num",">=",1000), ("d_num","<=",1913)]
    )
    df = df.dropna(subset=["lag_7","lag_14","lag_28","lag_56"]).reset_index(drop=True)
    tr = df[df["d_num"] <= 1885]; vl = df[df["d_num"] >= 1886]
    # Cast ALL cat cols to Categorical before Dataset construction (required by LightGBM)
    X_tr = tr[ALL_FEATURES].copy()
    X_vl = vl[ALL_FEATURES].copy()
    for col, dt in CAT_DTYPES.items():
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].astype(dt)
            X_vl[col] = X_vl[col].astype(dt)
    ds_tr = lgb.Dataset(X_tr, label=tr["sales"].values.astype(np.float32),
                        categorical_feature=CAT, free_raw_data=False)
    ds_vl = lgb.Dataset(X_vl, label=vl["sales"].values.astype(np.float32),
                        categorical_feature=CAT, reference=ds_tr, free_raw_data=False)
    t0 = time.time()
    model = lgb.train(PARAMS, ds_tr, num_boost_round=3000, valid_sets=[ds_vl],
                      callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(-1)])
    elapsed = time.time() - t0
    n_series = df["id"].nunique()
    print(f"BENCH {store}x{dept}: {n_series} series, {len(df):,} rows, "
          f"iter={model.best_iteration}, val={model.best_score['valid_0']['tweedie']:.4f}, "
          f"time={elapsed:.1f}s ({elapsed/60:.1f}min)")
