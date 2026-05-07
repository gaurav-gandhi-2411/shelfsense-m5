"""
Day 10: Per-store LightGBM models — one model per store (10 stores).

Each store model trains only on that store's series (~3,049 each), allowing
tree splits to specialise on store-level demand patterns: local events,
regional SNAP behaviour, store-specific product mix.

Hypothesis: per-store diversity improves private LB ensemble, same mechanism
as per-category diversity improved private LB from 0.81 → 0.71 (Day 7).

Training design matches Day 6/7 global model for apples-to-apples comparison:
  - Same 38 features
  - Same Tweedie loss
  - Optuna 15 trials per store (fast — ~3k series each)
  - Same val period d_1886-d_1913
  - Eval recursive: per-store model via predict_horizon() (same as Day 8 v2)

Blend: 0.6 × per-store + 0.4 × global (matches Day 7 winning ratio)
"""
from __future__ import annotations

import sys, os, time, json, pickle, gc, warnings
warnings.filterwarnings("ignore")

_SP = "C:/Users/gaura/anaconda3/Lib/site-packages"
if _SP not in sys.path:
    sys.path.insert(0, _SP)

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from features.hierarchy import CAT_DTYPES
from models.recursive_forecast_v2 import predict_horizon, _build_history_df, ALL_FEATURES, NUM_FEATURES, CAT_COLS
from evaluation.wrmsse import compute_wrmsse

print("=" * 60)
print("DAY 10 -- Per-Store LightGBM (10 models)")
print("=" * 60)

DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
MODELS    = os.path.join(PROJ_ROOT, "data", "models")
PS_DIR    = os.path.join(MODELS, "per_store")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
os.makedirs(PS_DIR, exist_ok=True)

LAST_TRAIN  = 1913
VAL_START   = 1886
FEAT_START  = 1000
HORIZON     = 28
VAL_END     = LAST_TRAIN + HORIZON       # 1941
EVAL_START  = VAL_END + 1               # 1942
EVAL_END    = VAL_END + HORIZON         # 1969
N_TRIALS    = 15

CAT_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]

# ── 1. Load training data ─────────────────────────────────────────────────────
print(f"\n[1] Loading training data (d_num {FEAT_START}-{LAST_TRAIN})...")
t0 = time.time()
df = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", FEAT_START), ("d_num", "<=", LAST_TRAIN)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)
df = df.dropna(subset=["lag_7","lag_14","lag_28","lag_56"]).reset_index(drop=True)
print(f"    Rows: {len(df):,}  ({time.time()-t0:.1f}s)")

print("\n    Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"    Loaded in {time.time()-t0:.1f}s")

STORES = sorted(df["store_id"].astype(str).unique().tolist())
print(f"\n    Stores: {STORES}")

# ── 2. Load single-step val parquet + global model ───────────────────────────
print(f"\n[2] Loading inference data (d_{LAST_TRAIN+1}-{VAL_END} for val single-step)...")
df_ss = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN + 1), ("d_num", "<=", VAL_END)],
    columns=["id", "d_num"] + NUM_FEATURES + CAT_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_ss.columns:
        df_ss[col] = df_ss[col].astype(dtype)
df_ss = df_ss.sort_values(["id", "d_num"]).reset_index(drop=True)

series_meta = (
    df_ss[["id", "cat_id", "dept_id", "store_id", "state_id"]]
    .drop_duplicates("id").sort_values("id").reset_index(drop=True)
)
series_ids = series_meta["id"].values
n_series   = len(series_ids)
print(f"    Val parquet: {df_ss.shape}  ({n_series:,} series)")

print("\n    Loading global model (Day 6)...")
with open(os.path.join(MODELS, "lgbm_best.pkl"), "rb") as f:
    model_global = pickle.load(f)

actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]

def score_preds(preds_mat, sids):
    sub  = sales_eval[sales_eval["id"].isin(sids)].set_index("id").reindex(sids).reset_index()
    acts = sub[actual_val_cols].values.astype(np.float32)
    s, _ = compute_wrmsse(
        preds=preds_mat, actuals=acts,
        sales_df=sub, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    return float(s)

# Global single-step baseline
preds_ss_flat = np.clip(model_global.predict(df_ss[ALL_FEATURES]), 0, None).astype(np.float32)
val_preds_global = preds_ss_flat.reshape(n_series, HORIZON)
wrmsse_global = score_preds(val_preds_global, series_ids)
print(f"    Global single-step WRMSSE (reference): {wrmsse_global:.4f}")

# ── 3. Optuna + train per-store models ────────────────────────────────────────
print(f"\n[3] Training per-store models ({len(STORES)} stores, {N_TRIALS} Optuna trials each)...")

store_models  = {}
store_params  = {}
store_scores  = {}

for store in STORES:
    pkl_path = os.path.join(PS_DIR, f"store_{store}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            store_models[store] = pickle.load(f)
        print(f"\n  {store}: loaded from checkpoint")
        continue

    print(f"\n  {store}:")
    store_mask = df["store_id"].astype(str) == store
    df_s = df[store_mask].copy()

    train_mask = (df_s["d_num"] >= FEAT_START) & (df_s["d_num"] <= LAST_TRAIN)
    val_mask   = (df_s["d_num"] >= VAL_START)  & (df_s["d_num"] <= LAST_TRAIN)

    X_tr_full = df_s.loc[train_mask, ALL_FEATURES]
    y_tr_full = df_s.loc[train_mask, "sales"].astype(np.float32)
    X_vl      = df_s.loc[val_mask,   ALL_FEATURES]
    y_vl      = df_s.loc[val_mask,   "sales"].astype(np.float32)
    print(f"    Train: {len(X_tr_full):,}  Val: {len(X_vl):,}")

    # Optuna on training subset for speed
    opt_mask = (df_s["d_num"] >= 1600) & (df_s["d_num"] <= LAST_TRAIN)
    X_opt = df_s.loc[opt_mask & train_mask, ALL_FEATURES]
    y_opt = df_s.loc[opt_mask & train_mask, "sales"].astype(np.float32)

    ds_opt = lgb.Dataset(X_opt, label=y_opt, categorical_feature=CAT_FEATURES, free_raw_data=False)
    ds_vl_opt = lgb.Dataset(X_vl, label=y_vl, categorical_feature=CAT_FEATURES,
                            reference=ds_opt, free_raw_data=False)

    def make_objective(ds_tr, ds_val):
        def objective(trial):
            p = {
                "objective":              "tweedie",
                "metric":                 "tweedie",
                "verbose":               -1,
                "seed":                   42,
                "num_threads":            0,
                "bagging_freq":           1,
                "learning_rate":          trial.suggest_categorical("lr",        [0.025, 0.05, 0.075, 0.1]),
                "num_leaves":             trial.suggest_categorical("num_leaves", [32, 64, 128, 256]),
                "min_data_in_leaf":       trial.suggest_categorical("min_data",   [10, 20, 50, 100]),
                "feature_fraction":       trial.suggest_categorical("feat_frac",  [0.5, 0.7, 0.8, 0.9]),
                "bagging_fraction":       trial.suggest_categorical("bag_frac",   [0.5, 0.7, 0.8, 0.9]),
                "lambda_l2":              trial.suggest_categorical("l2",         [0.0, 0.1, 0.5, 1.0]),
                "tweedie_variance_power": trial.suggest_float("tvp", 1.0, 1.9),
            }
            m = lgb.train(
                p, ds_tr, num_boost_round=500,
                valid_sets=[ds_val],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            return m.best_score["valid_0"]["tweedie"]
        return objective

    t0 = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(ds_opt, ds_vl_opt), n_trials=N_TRIALS, show_progress_bar=False)
    bp = study.best_params
    print(f"    Optuna done in {time.time()-t0:.0f}s  best={study.best_value:.4f}")
    print(f"    Params: lr={bp.get('lr')} leaves={bp.get('num_leaves')} tvp={bp.get('tvp',0):.3f}")

    del ds_opt, ds_vl_opt, X_opt, y_opt
    gc.collect()

    best_params = {
        "objective":              "tweedie",
        "metric":                 "tweedie",
        "verbose":               -1,
        "num_threads":            0,
        "seed":                   42,
        "bagging_freq":           1,
        "learning_rate":          bp.get("lr",        0.05),
        "num_leaves":             bp.get("num_leaves", 64),
        "min_data_in_leaf":       bp.get("min_data",   20),
        "feature_fraction":       bp.get("feat_frac",  0.9),
        "bagging_fraction":       bp.get("bag_frac",   0.9),
        "lambda_l2":              bp.get("l2",         1.0),
        "tweedie_variance_power": bp.get("tvp",        1.5),
    }

    ds_tr = lgb.Dataset(X_tr_full, label=y_tr_full, categorical_feature=CAT_FEATURES, free_raw_data=False)
    ds_vl = lgb.Dataset(X_vl,      label=y_vl,      categorical_feature=CAT_FEATURES,
                        reference=ds_tr, free_raw_data=False)

    t0 = time.time()
    model = lgb.train(
        best_params, ds_tr,
        num_boost_round=3000,
        valid_sets=[ds_vl],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(500)],
    )
    elapsed = time.time() - t0
    val_metric = model.best_score["valid_0"]["tweedie"]
    print(f"    Train done in {elapsed:.0f}s  iter={model.best_iteration}  val={val_metric:.4f}")

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    store_models[store]  = model
    store_params[store]  = {**bp, "best_iter": model.best_iteration, "val_tweedie": val_metric}
    store_scores[store]  = {"val_tweedie": val_metric, "best_iter": model.best_iteration, "train_s": elapsed}

    del ds_tr, ds_vl, X_tr_full, y_tr_full, X_vl, y_vl, df_s
    gc.collect()

print(f"\n    All {len(STORES)} store models ready.")

# ── 4. Val WRMSSE single-step per-store ───────────────────────────────────────
print(f"\n[4] Val WRMSSE — per-store single-step (d_{LAST_TRAIN+1}-{VAL_END})...")
val_preds_perstore = np.zeros((n_series, HORIZON), dtype=np.float32)

for store in STORES:
    store_mask = df_ss["store_id"].astype(str) == store
    store_df   = df_ss[store_mask].sort_values(["id", "d_num"])
    store_sids = store_df.groupby("id", sort=True).first().reset_index()["id"].values
    flat       = np.clip(store_models[store].predict(store_df[ALL_FEATURES]), 0, None).astype(np.float32)
    mat        = flat.reshape(len(store_sids), HORIZON)
    idx        = np.searchsorted(series_ids, store_sids)
    val_preds_perstore[idx] = mat

wrmsse_perstore = score_preds(val_preds_perstore, series_ids)
val_preds_blend = 0.6 * val_preds_perstore + 0.4 * val_preds_global
wrmsse_blend_val = score_preds(val_preds_blend, series_ids)

print(f"    Global single-step WRMSSE:     {wrmsse_global:.4f}")
print(f"    Per-store single-step WRMSSE:  {wrmsse_perstore:.4f}")
print(f"    Blend 0.6/0.4 WRMSSE:          {wrmsse_blend_val:.4f}")

# ── 5. Eval recursive per-store (d_1942-1969) ─────────────────────────────────
print(f"\n[5] Eval recursive — per-store (d_{EVAL_START}-{EVAL_END})...")
history_eval = _build_history_df(sales_eval, series_ids, last_day=VAL_END)
print(f"    History: d_num range [{history_eval['d_num'].min()}, {history_eval['d_num'].max()}]")

eval_preds_perstore = np.zeros((n_series, HORIZON), dtype=np.float32)

for store in STORES:
    store_mask_meta = series_meta["store_id"].astype(str) == store
    store_indices   = np.where(store_mask_meta)[0]
    store_sids_eval = series_ids[store_indices]

    store_history = history_eval[history_eval["id"].isin(store_sids_eval)]
    print(f"\n    {store} ({len(store_sids_eval):,} series):")
    t0 = time.time()
    store_eval_preds, _ = predict_horizon(
        store_models[store], store_history, calendar_df, prices_df,
        days_out=HORIZON, cat_dtypes=CAT_DTYPES, verbose=True,
    )
    eval_preds_perstore[store_indices] = store_eval_preds
    print(f"    {store} done in {time.time()-t0:.1f}s")

# ── 5b. Global recursive eval for blend ──────────────────────────────────────
print(f"\n[5b] Global recursive eval (for blend)...")
t0 = time.time()
eval_preds_global, _ = predict_horizon(
    model_global, history_eval, calendar_df, prices_df,
    days_out=HORIZON, cat_dtypes=CAT_DTYPES, verbose=True,
)
print(f"    Global recursive done in {time.time()-t0:.1f}s")

eval_preds_blend = 0.6 * eval_preds_perstore + 0.4 * eval_preds_global

# ── 6. Build Kaggle submissions ───────────────────────────────────────────────
print(f"\n[6] Building submissions...")
sn28 = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))
f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]

def build_submission(val_mat, eval_mat, sids, fname):
    base     = sn28.copy().set_index("id")
    val_ids  = [s.replace("_evaluation", "_validation") for s in sids]
    val_df   = pd.DataFrame(val_mat, columns=f_cols)
    val_df.insert(0, "id", val_ids)
    val_idx  = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = val_df.set_index("id").loc[val_idx]
    eval_df  = pd.DataFrame(eval_mat, columns=f_cols)
    eval_df.insert(0, "id", list(sids))
    eval_idx = pd.Index(sids).intersection(base.index)
    base.loc[eval_idx] = eval_df.set_index("id").loc[eval_idx]
    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"    Saved: {fname}  ({len(val_idx)} val + {len(eval_idx)} eval rows)")
    return path

build_submission(val_preds_perstore, eval_preds_perstore, series_ids, "per_store_only.csv")
build_submission(val_preds_blend,    eval_preds_blend,    series_ids, "per_store_blend.csv")

# ── 7. Save results JSON ──────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results = {
    "val_wrmsse": {
        "global_single_step":   wrmsse_global,
        "perstore_single_step": wrmsse_perstore,
        "blend_0.6_0.4":        wrmsse_blend_val,
    },
    "store_training": make_serial(store_scores),
    "store_params":   make_serial(store_params),
}

scores_path = os.path.join(REPORTS, "day10_scores.json")
with open(scores_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved: {scores_path}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DAY 10 SUMMARY")
print("=" * 60)
print(f"\nVal WRMSSE (single-step, 30490 series):")
print(f"  Global:    {wrmsse_global:.4f}")
print(f"  Per-store: {wrmsse_perstore:.4f}  ({'better' if wrmsse_perstore < wrmsse_global else 'worse'} than global)")
print(f"  Blend:     {wrmsse_blend_val:.4f}")
print(f"\nPer-store Optuna params:")
for store in STORES:
    p = store_params.get(store, {})
    if p:
        print(f"  {store}: lr={p.get('lr')}  leaves={p.get('num_leaves')}  tvp={p.get('tvp',0):.3f}  iter={p.get('best_iter')}")
    else:
        print(f"  {store}: loaded from checkpoint")
print(f"\nSubmissions built:")
print(f"  submissions/per_store_only.csv  — per-store val + recursive eval")
print(f"  submissions/per_store_blend.csv — 0.6×per-store + 0.4×global blend")
print(f"\nTo submit:")
print(f'  kaggle competitions submit -c m5-forecasting-accuracy -f submissions/per_store_only.csv -m "Day 10: per-store LightGBM (10 models)"')
print(f'  kaggle competitions submit -c m5-forecasting-accuracy -f submissions/per_store_blend.csv -m "Day 10: blend 0.6x per-store + 0.4x global recursive"')
print("\nDone!")
