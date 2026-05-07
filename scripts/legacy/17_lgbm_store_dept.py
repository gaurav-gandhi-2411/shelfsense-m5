"""
WS2.5 - Store x Dept LightGBM (70 slices, single-horizon recursive)

One Tweedie tvp=1.3 model per store x dept slice (10 stores x 7 depts = 70).
28-step recursive forecast via recursive_forecast_v2.predict_horizon.

Key design:
  - Optuna 10 trials/slice: learning_rate, num_leaves, min_data_in_leaf,
    feature_fraction, bagging_fraction. TVP fixed at 1.3.
  - Cache: data/models/store_dept/lgbm_SD_{store}_{dept}_p{hash8}.pkl
    Hash covers objective, tvp, data range, n_trials, feature-set version.
    Change any design constant -> hash changes -> full retrain.
  - WARN flag: slice val_tweedie > 2x dept-wide average (still included in ensemble).
  - Fallback: any uncovered series use tvp=1.3 predictions in full-catalogue submit.
  - INTERPRETATION CAVEAT (printed in final summary):
    SD = recursive (8-12% compounding gap).
    tvp=1.3 = multi-horizon direct (no compounding).
    Private LB delta between the two conflates slicing benefit with recursive penalty.
    Cannot fully separate without a 28-model SD variant (not done).

After all slices:
  A. Per-slice val WRMSSE table (same-origin, d_1913)
  B. Full-catalogue same-origin val WRMSSE
  C. 50-trial Optuna ensemble {SD, tvp=1.3}; report weights
  D. Kaggle submit: SD standalone + ensemble if not degenerate (neither weight >0.85)
  E. Final results summary + JSON
"""
from __future__ import annotations
import sys, os, time, json, pickle, hashlib, warnings, subprocess
warnings.filterwarnings("ignore")

_SP = "C:/Users/gaura/anaconda3/Lib/site-packages"
if _SP not in sys.path:
    sys.path.insert(0, _SP)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from features.hierarchy import CAT_DTYPES
from evaluation.wrmsse import compute_wrmsse
from models.recursive_forecast_v2 import predict_horizon

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ, "data", "processed", "features")
PREDS_DIR = os.path.join(PROJ, "data", "predictions")
REPORTS   = os.path.join(PROJ, "reports")
SUBS      = os.path.join(PROJ, "submissions")
MODELS_SD = os.path.join(PROJ, "data", "models", "store_dept")
for _d in [MODELS_SD, PREDS_DIR, SUBS]:
    os.makedirs(_d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LAST_TRAIN   = 1913
VAL_START    = 1886
FEAT_START   = 1000
HORIZON      = 28
N_OPTUNA     = 10
HISTORY_DAYS = 200  # must match recursive_forecast_v2.HISTORY_DAYS

STORES = ["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"]
DEPTS  = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2"]

actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]
f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]

# ── Design hash (cache key) ───────────────────────────────────────────────────
# Change ANY of these constants and the hash flips -> full retrain, no stale models.
_DESIGN = {
    "objective":      "tweedie",
    "tvp":            1.3,
    "feat_start":     FEAT_START,
    "last_train":     LAST_TRAIN,
    "val_start":      VAL_START,
    "n_optuna":       N_OPTUNA,
    "feature_set":    "v1_38num_4cat",
}
DESIGN_HASH = hashlib.md5(json.dumps(_DESIGN, sort_keys=True).encode()).hexdigest()[:8]

# ── Feature list (identical to scripts/12 and 16 — no yearly lags) ───────────
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
CAT_FEATURES = ["cat_id","dept_id","store_id","state_id"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

BASE_PARAMS = {
    "objective":              "tweedie",
    "metric":                 "tweedie",
    "verbose":               -1,
    "num_threads":            0,
    "bagging_freq":           1,
    "tweedie_variance_power": 1.3,
    "seed":                   42,
}

print("=" * 65)
print("WS2.5 -- Store x Dept LightGBM (70 slices, recursive)")
print("=" * 65)
print(f"\n  Design hash:   {DESIGN_HASH}")
print(f"  Slices:        {len(STORES)*len(DEPTS)} ({len(STORES)} stores x {len(DEPTS)} depts)")
print(f"  Optuna:        {N_OPTUNA} trials/slice (LR, leaves, min_leaf, ff, bf)")
print(f"  Cache dir:     data/models/store_dept/")
print(f"  Fallback:      tvp=1.3 for any uncovered series")


# ── Helper: build history_df for predict_horizon ─────────────────────────────
def _build_hist(sales_df, series_ids, last_day):
    """Long-format sales history for predict_horizon (HISTORY_DAYS days per series)."""
    first_day = last_day - HISTORY_DAYS + 1
    day_cols = [f"d_{d}" for d in range(first_day, last_day + 1)
                if f"d_{d}" in sales_df.columns]
    meta = ["id","item_id","cat_id","dept_id","store_id","state_id"]
    sub = (sales_df[sales_df["id"].isin(series_ids)]
           .set_index("id").reindex(list(series_ids)).reset_index())
    df = sub[meta + day_cols].melt(id_vars=meta, var_name="d", value_name="sales")
    df["d_num"] = df["d"].str.replace("d_","",regex=False).astype(np.int32)
    df["sales"] = df["sales"].fillna(0.0).astype(np.float32)
    return df.drop(columns=["d"])


# ── Helper: Optuna + train, returns best model seen across all trials ─────────
def train_slice_optuna(df_tr, df_vl, n_trials=N_OPTUNA):
    """
    10-trial Optuna sweep then return best model.
    Dataset objects are built once and reused across trials (free_raw_data=False).
    """
    X_tr = df_tr[ALL_FEATURES]
    y_tr = df_tr["sales"].values.astype(np.float32)
    X_vl = df_vl[ALL_FEATURES]
    y_vl = df_vl["sales"].values.astype(np.float32)

    ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_FEATURES, free_raw_data=False)
    ds_vl = lgb.Dataset(X_vl, label=y_vl, categorical_feature=CAT_FEATURES,
                        reference=ds_tr, free_raw_data=False)

    best = {"val": float("inf"), "model": None, "params": None, "iter": 0}

    def _obj(trial):
        params = {
            **BASE_PARAMS,
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        }
        model = lgb.train(
            params, ds_tr, num_boost_round=3000,
            valid_sets=[ds_vl],
            callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(-1)],
        )
        val = model.best_score["valid_0"]["tweedie"]
        if val < best["val"]:
            best.update(val=val, model=model, params=dict(trial.params), iter=model.best_iteration)
        return val

    study = optuna.create_study(direction="minimize")
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)
    return best["model"], best["val"], best["params"], best["iter"]


def _make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: _make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_make_serial(x) for x in obj]
    return obj


# ── 1. Load raw CSVs once ─────────────────────────────────────────────────────
print("\n[1] Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"    Loaded in {time.time()-t0:.1f}s")

# Canonical series order (evaluation IDs, matches sales_eval row order)
all_series_eval = sales_eval["id"].values  # shape (30490,)
sales_eval_idx  = sales_eval.set_index("id")


# ── 2. Training loop ──────────────────────────────────────────────────────────
print(f"\n[2] Training {len(STORES)*len(DEPTS)} store x dept slices...")
slice_results = {}   # (store, dept) -> result dict (no model object after save)
t_wall_start  = time.time()
n_cached = 0
n_trained = 0

for s_idx, store in enumerate(STORES):
    print(f"\n  [Store {s_idx+1}/{len(STORES)}: {store}]", flush=True)

    # Load full store parquet once; filter per dept inside the dept loop
    t_load = time.time()
    _cols = list(dict.fromkeys(["id","item_id","d_num","sales"] + NUM_FEATURES + CAT_FEATURES))
    store_df = pd.read_parquet(
        os.path.join(FEAT_DIR, f"store_{store}.parquet"),
        columns=_cols,
        filters=[("d_num",">=",FEAT_START), ("d_num","<=",LAST_TRAIN)],
    )
    for col, dt in CAT_DTYPES.items():
        if col in store_df.columns:
            store_df[col] = store_df[col].astype(dt)
    store_df = store_df.dropna(subset=["lag_7","lag_14","lag_28","lag_56"]).reset_index(drop=True)
    print(f"    Loaded {len(store_df):,} rows in {time.time()-t_load:.1f}s", flush=True)

    for dept in DEPTS:
        pkl_name = f"lgbm_SD_{store}_{dept}_p{DESIGN_HASH}.pkl"
        pkl_path = os.path.join(MODELS_SD, pkl_name)

        # ── cache hit ─────────────────────────────────────────────────────────
        if os.path.exists(pkl_path):
            print(f"    {store}x{dept}: cached [{pkl_name}]", flush=True)
            with open(pkl_path, "rb") as _f:
                cached = pickle.load(_f)
            # Strip model object from in-memory result (save RAM); model stays on disk
            slice_results[(store, dept)] = {k: v for k, v in cached.items() if k != "model"}
            n_cached += 1
            continue

        # ── filter to dept ────────────────────────────────────────────────────
        df = store_df[store_df["dept_id"] == dept].copy()
        if len(df) == 0:
            print(f"    {store}x{dept}: SKIP — no rows", flush=True)
            continue
        n_series = df["id"].nunique()
        series_ids = sorted(df["id"].unique())

        df_tr = df[df["d_num"] <= VAL_START - 1]
        df_vl = df[df["d_num"] >= VAL_START]

        # ── Optuna + train ────────────────────────────────────────────────────
        t_slice = time.time()
        model, val_tweedie, best_params, best_iter = train_slice_optuna(df_tr, df_vl)
        train_time = time.time() - t_slice

        # ── Recursive predictions ─────────────────────────────────────────────
        hist_val  = _build_hist(sales_eval, series_ids, last_day=LAST_TRAIN)
        val_preds, _ = predict_horizon(
            model, hist_val, calendar_df, prices_df,
            days_out=HORIZON, verbose=False,
        )

        hist_eval = _build_hist(sales_eval, series_ids, last_day=LAST_TRAIN + HORIZON)
        eval_preds, _ = predict_horizon(
            model, hist_eval, calendar_df, prices_df,
            days_out=HORIZON, verbose=False,
        )

        # ── Per-slice val WRMSSE ─────────────────────────────────────────────
        try:
            sub_sales = (sales_eval[sales_eval["id"].isin(series_ids)]
                         .set_index("id").reindex(series_ids).reset_index())
            actuals_sl = sub_sales[actual_val_cols].values.astype(np.float32)
            slice_wrmsse, _ = compute_wrmsse(
                val_preds, actuals_sl, sub_sales, prices_df, calendar_df, LAST_TRAIN)
            slice_wrmsse = float(slice_wrmsse)
        except Exception as exc:
            print(f"      WRMSSE scoring failed: {exc}")
            slice_wrmsse = float("nan")

        result = {
            "store":        store,
            "dept":         dept,
            "n_series":     n_series,
            "val_tweedie":  float(val_tweedie),
            "val_wrmsse":   slice_wrmsse,
            "best_iter":    int(best_iter),
            "best_params":  best_params,
            "train_time":   float(train_time),
            "val_preds":    val_preds.astype(np.float32),
            "eval_preds":   eval_preds.astype(np.float32),
            "series_ids":   series_ids,
            "warn":         False,
        }

        # ── Save pkl (model + result dict) ────────────────────────────────────
        with open(pkl_path, "wb") as _f:
            pickle.dump({**result, "model": model}, _f)

        slice_results[(store, dept)] = result  # no model object in RAM
        n_trained += 1

        elapsed = time.time() - t_wall_start
        n_done = n_cached + n_trained
        eta_min = (elapsed / n_done) * (len(STORES)*len(DEPTS) - n_done) / 60 if n_done else 0
        print(f"    {store}x{dept}: n={n_series}, iter={best_iter}, "
              f"val_tweedie={val_tweedie:.4f}, wrmsse={slice_wrmsse:.4f}, "
              f"time={train_time:.1f}s  [ETA {eta_min:.0f}min]", flush=True)

total_train_time = time.time() - t_wall_start
print(f"\n  Training complete: {n_trained} trained, {n_cached} from cache. "
      f"Total: {total_train_time/60:.1f}min")


# ── 3. WARN flags ─────────────────────────────────────────────────────────────
print("\n[3] WARN flags (slice val_tweedie > 2x dept-wide average)...")
dept_avg = {}
for dept in DEPTS:
    vals = [slice_results[(s,dept)]["val_tweedie"]
            for s in STORES if (s,dept) in slice_results]
    dept_avg[dept] = float(np.mean(vals)) if vals else 0.0

warn_list = []
for (store, dept), r in slice_results.items():
    threshold = 2.0 * dept_avg[dept]
    if r["val_tweedie"] > threshold:
        r["warn"] = True
        msg = (f"WARN {store}x{dept}: val_tweedie={r['val_tweedie']:.4f} "
               f"> 2x dept_avg={dept_avg[dept]:.4f} (threshold={threshold:.4f})")
        warn_list.append(msg)
        print(f"  {msg}")

if not warn_list:
    print("  No WARN flags raised.")


# ── 4A. Per-slice val WRMSSE table ────────────────────────────────────────────
print("\n[4A] Per-slice val WRMSSE table (same-origin d_1913):")
print(f"\n  {'Store':<8} {'Dept':<14} {'Series':>7} {'WRMSSE':>8} {'Tweedie':>9} {'Iter':>6} {'Warn'}")
print(f"  {'-'*8} {'-'*14} {'-'*7} {'-'*8} {'-'*9} {'-'*6} {'-'*4}")
dept_wrmsse_list = {d: [] for d in DEPTS}
for store in STORES:
    for dept in DEPTS:
        key = (store, dept)
        if key not in slice_results:
            continue
        r = slice_results[key]
        w = "WARN" if r.get("warn") else ""
        print(f"  {store:<8} {dept:<14} {r['n_series']:>7} {r['val_wrmsse']:>8.4f} "
              f"{r['val_tweedie']:>9.4f} {r['best_iter']:>6} {w}")
        if not np.isnan(r["val_wrmsse"]):
            dept_wrmsse_list[dept].append(r["val_wrmsse"])

print(f"\n  Dept averages:")
for dept in DEPTS:
    vs = dept_wrmsse_list[dept]
    if vs:
        print(f"    {dept:<14}: mean WRMSSE = {np.mean(vs):.4f}  "
              f"(range {min(vs):.4f}–{max(vs):.4f})")


# ── 4B. Full-catalogue assembly (with tvp=1.3 fallback) ──────────────────────
print("\n[4B] Assembling full-catalogue predictions (30,490 series)...")
sd_val_lookup  = {}  # eval_id -> (28,) array
sd_eval_lookup = {}

for (store, dept), r in slice_results.items():
    for i, sid in enumerate(r["series_ids"]):
        sd_val_lookup[sid]  = r["val_preds"][i]
        sd_eval_lookup[sid] = r["eval_preds"][i]

# Load tvp=1.3 fallback predictions
tvp13_val_df  = pd.read_parquet(
    os.path.join(PREDS_DIR, "lgbm_tvp_1p3_val.parquet")).set_index("id")
tvp13_eval_df = pd.read_parquet(
    os.path.join(PREDS_DIR, "lgbm_tvp_1p3_eval.parquet")).set_index("id")

n_fallback = 0
sd_val_mat  = np.zeros((len(all_series_eval), HORIZON), dtype=np.float32)
sd_eval_mat = np.zeros((len(all_series_eval), HORIZON), dtype=np.float32)

for i, sid in enumerate(all_series_eval):
    if sid in sd_val_lookup:
        sd_val_mat[i]  = sd_val_lookup[sid]
        sd_eval_mat[i] = sd_eval_lookup[sid]
    else:
        sd_val_mat[i]  = tvp13_val_df.loc[sid, f_cols].values.astype(np.float32)
        sd_eval_mat[i] = tvp13_eval_df.loc[sid, f_cols].values.astype(np.float32)
        n_fallback += 1

print(f"  SD coverage: {len(all_series_eval)-n_fallback}/{len(all_series_eval)} series  "
      f"({n_fallback} fallback to tvp=1.3)")


# ── 5. Full-catalogue same-origin val WRMSSE ──────────────────────────────────
print("\n[5] Full-catalogue same-origin val WRMSSE (d_1913 origin)...")
sales_eval_reindexed = (sales_eval.set_index("id")
                        .reindex(all_series_eval).reset_index())
actuals_full = sales_eval_reindexed[actual_val_cols].values.astype(np.float32)

sd_full_wrmsse, _ = compute_wrmsse(
    sd_val_mat, actuals_full,
    sales_eval_reindexed, prices_df, calendar_df, LAST_TRAIN,
)
sd_full_wrmsse = float(sd_full_wrmsse)
print(f"  SD full-catalogue val WRMSSE:  {sd_full_wrmsse:.4f}")
print(f"  tvp=1.3 baseline (same-origin): 0.6860")
print(f"  Delta:                          {sd_full_wrmsse - 0.6860:+.4f}")


# ── 6. Optuna ensemble {SD, tvp=1.3} — 50 trials ─────────────────────────────
print("\n[6] Optuna ensemble {SD full-catalogue, tvp=1.3} -- 50 trials...")
tvp13_val_mat = tvp13_val_df.reindex(all_series_eval)[f_cols].values.astype(np.float32)

def _ens_obj(trial):
    w = trial.suggest_float("w_sd", 0.0, 1.0)
    blended = w * sd_val_mat + (1.0 - w) * tvp13_val_mat
    wrmsse, _ = compute_wrmsse(
        blended, actuals_full,
        sales_eval_reindexed, prices_df, calendar_df, LAST_TRAIN,
    )
    return float(wrmsse)

ens_study = optuna.create_study(direction="minimize")
ens_study.optimize(_ens_obj, n_trials=50)
best_w_sd      = float(ens_study.best_params["w_sd"])
ens_val_wrmsse = float(ens_study.best_value)
degenerate     = max(best_w_sd, 1.0 - best_w_sd) > 0.85

print(f"  Best ensemble val WRMSSE: {ens_val_wrmsse:.4f}")
print(f"  Weights: SD={best_w_sd:.3f}, tvp13={1-best_w_sd:.3f}")
print(f"  Degenerate (one weight >0.85): {degenerate}")


# ── 7. Build submission CSVs ───────────────────────────────────────────────────
print("\n[7] Building submission CSVs...")
val_ids  = [s.replace("_evaluation","_validation") for s in all_series_eval]
eval_ids = list(all_series_eval)

# 7a. SD standalone
sd_sub = pd.DataFrame({"id": val_ids + eval_ids})
for h in range(1, HORIZON + 1):
    sd_sub[f"F{h}"] = np.concatenate([sd_val_mat[:, h-1], sd_eval_mat[:, h-1]])
sd_sub_path = os.path.join(SUBS, "lgbm_store_dept.csv")
sd_sub.to_csv(sd_sub_path, index=False)
print(f"  Saved SD standalone: {sd_sub_path}")

# 7b. Ensemble (only if not degenerate)
ens_sub_path = None
if not degenerate:
    tvp13_eval_mat = tvp13_eval_df.reindex(all_series_eval)[f_cols].values.astype(np.float32)
    ens_val_mat  = best_w_sd * sd_val_mat  + (1.0 - best_w_sd) * tvp13_val_mat
    ens_eval_mat = best_w_sd * sd_eval_mat + (1.0 - best_w_sd) * tvp13_eval_mat
    ens_sub = pd.DataFrame({"id": val_ids + eval_ids})
    for h in range(1, HORIZON + 1):
        ens_sub[f"F{h}"] = np.concatenate([ens_val_mat[:, h-1], ens_eval_mat[:, h-1]])
    ens_sub_path = os.path.join(SUBS, "lgbm_store_dept_ensemble.csv")
    ens_sub.to_csv(ens_sub_path, index=False)
    print(f"  Saved ensemble: {ens_sub_path}")
else:
    print(f"  Ensemble degenerate (w_SD={best_w_sd:.3f}) -- not built.")

# Save full-catalogue predictions as parquets for future ensembling
sd_val_pq = pd.DataFrame({"id": list(all_series_eval)})
sd_eval_pq = pd.DataFrame({"id": list(all_series_eval)})
for h in range(1, HORIZON + 1):
    sd_val_pq[f"F{h}"]  = sd_val_mat[:, h-1]
    sd_eval_pq[f"F{h}"] = sd_eval_mat[:, h-1]
sd_val_pq.to_parquet(os.path.join(PREDS_DIR, "lgbm_store_dept_val.parquet"),  index=False)
sd_eval_pq.to_parquet(os.path.join(PREDS_DIR, "lgbm_store_dept_eval.parquet"), index=False)
print(f"  Saved prediction parquets (lgbm_store_dept_{{val,eval}}.parquet)")


# ── 8. Kaggle submissions ─────────────────────────────────────────────────────
print("\n[8] Submitting to Kaggle...")

def _kaggle_submit(csv_path, message):
    cmd = (f'kaggle competitions submit -c m5-forecasting-accuracy '
           f'-f "{csv_path}" -m "{message}"')
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if res.returncode == 0:
        print(f"  OK: {message}")
        if res.stdout.strip():
            print(f"     {res.stdout.strip()}")
    else:
        print(f"  FAIL: {message}")
        print(f"     {res.stderr.strip()}")
    return res.returncode == 0

ok_sd  = _kaggle_submit(sd_sub_path,
    f"SD: store-dept 70 slices recursive tvp=1.3 (same-origin val={sd_full_wrmsse:.4f})")
ok_ens = False
if ens_sub_path:
    ok_ens = _kaggle_submit(ens_sub_path,
        f"Ens SD+tvp13: w_SD={best_w_sd:.2f} val={ens_val_wrmsse:.4f}")


# ── 9. Save JSON ───────────────────────────────────────────────────────────────
results_json = {
    "design_hash":         DESIGN_HASH,
    "design":              _DESIGN,
    "n_slices_trained":    n_trained,
    "n_slices_cached":     n_cached,
    "total_train_min":     round(total_train_time / 60, 2),
    "sd_full_val_wrmsse":  sd_full_wrmsse,
    "tvp13_baseline":      0.6860,
    "delta_vs_tvp13":      round(sd_full_wrmsse - 0.6860, 4),
    "ens_val_wrmsse":      ens_val_wrmsse,
    "ens_weight_sd":       best_w_sd,
    "ens_degenerate":      degenerate,
    "n_fallback_series":   n_fallback,
    "warn_list":           warn_list,
    "dept_avg_val_tweedie": {k: round(v, 4) for k, v in dept_avg.items()},
    "per_slice": {
        f"{store}_{dept}": _make_serial({
            "n_series":    r["n_series"],
            "val_wrmsse":  r["val_wrmsse"],
            "val_tweedie": r["val_tweedie"],
            "best_iter":   r["best_iter"],
            "train_time":  r["train_time"],
            "warn":        r.get("warn", False),
            "best_params": r["best_params"],
        })
        for (store, dept), r in slice_results.items()
    },
}
json_path = os.path.join(REPORTS, "day17_store_dept_scores.json")
with open(json_path, "w") as _f:
    json.dump(_make_serial(results_json), _f, indent=2)
print(f"\n  Saved {json_path}")


# ── 10. Final results summary ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"\n  Slices:         {n_trained} trained, {n_cached} from cache, {n_fallback} fallback series")
print(f"  Training time:  {total_train_time/60:.1f} min total")
print(f"\n  SD full-catalogue val WRMSSE:  {sd_full_wrmsse:.4f}")
print(f"  tvp=1.3 baseline:              0.6860")
print(f"  Delta:                         {sd_full_wrmsse - 0.6860:+.4f}  "
      f"({'better' if sd_full_wrmsse < 0.6860 else 'worse'})")
print(f"\n  Ensemble val WRMSSE:    {ens_val_wrmsse:.4f}")
print(f"  Ensemble weights:       SD={best_w_sd:.3f}  tvp13={1-best_w_sd:.3f}")
print(f"  Ensemble degenerate:    {degenerate}")
print(f"\n  Kaggle submits:  SD={'OK' if ok_sd else 'FAIL'}  "
      f"Ensemble={'OK' if ok_ens else ('SKIP' if not ens_sub_path else 'FAIL')}")
if warn_list:
    print(f"\n  WARN flags ({len(warn_list)} slices):")
    for w in warn_list:
        print(f"    {w}")

print(f"""
  INTERPRETATION CAVEAT
  ----------------------
  Store-dept models use RECURSIVE forecasting (predict_horizon v2).
  This introduces 8-12% error compounding over 28 steps on sparse M5 data.
  The tvp=1.3 baseline uses MULTI-HORIZON DIRECT training (no compounding).

  If SD loses to tvp=1.3 on private LB, the loss could be from:
    (a) slicing penalty  -- smaller training sets lose cross-series regularization
    (b) recursive penalty -- compounding lag errors over 28 steps
    (c) both
  These cannot be separated without a 28-model per-slice variant (not implemented).
  The M5 1st-place result (220 LightGBM models) used recursive; the penalty was
  apparently outweighed by slice diversity at that scale.
""")
print("Done!")
