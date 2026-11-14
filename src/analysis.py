"""Load the official snapshot, audit it, validate models and export evidence.

Run from the repository root: python -m src.analysis
All metrics and figures are calculated from the bundled CSV, never hardcoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .features import FUEL, NUMERIC, FuelInteractionFeatures

ROOT = Path(__file__).resolve().parents[1]
TARGET = "CO2 emissions (g/km)"
SEED = 42
SOURCE_URL = ("https://open.canada.ca/data/dataset/98f1a129-f628-4ce4-b24d-"
              "6f16bf24dd64/resource/d589f2bc-9a85-4f65-be2f-20f17debfcb1/"
              "download/my2025-fuel-consumption-ratings.csv")
SNAPSHOT_SHA256 = "46088cd4853987ef7a684f0b0af86e93e83dfaa378607a4750958c996248639c"
KEEP = ["Model year", "Make", "Model", "Vehicle class", *NUMERIC, FUEL, TARGET]
FUEL_LABELS = {"X": "Regular gasoline", "Z": "Premium gasoline", "D": "Diesel", "E": "Ethanol (E85)"}
COLORS = {"X": "#168a87", "Z": "#244b70", "D": "#d8872d", "E": "#9b548f"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_and_audit(path: Path):
    """Keep original bytes immutable and retain a source-row ID for tracing."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SNAPSHOT_SHA256:
        raise ValueError("CSV hash differs from the published project snapshot. Restore data/raw first.")
    # CP1252 preserves accented manufacturer/model names in the source file.
    original = pd.read_csv(path, encoding="cp1252")
    full_memory = int(original.memory_usage(deep=True).sum())
    selected_memory = int(original[KEEP].memory_usage(deep=True).sum())
    raw_missing = {k: int(v) for k, v in original.isna().sum().items()}
    duplicates = int(original.duplicated().sum())
    # Drop exact full-row duplicates only; distinct trims/specifications remain.
    df = original.loc[~original.duplicated(), KEEP].copy()
    df.insert(0, "source_row", df.index.to_numpy() + 2)  # CSV header is row 1.
    for col in ["Make", "Model", "Vehicle class", FUEL]:
        df[col] = df[col].astype("string").str.strip()
    for col in [*NUMERIC, TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid = df[TARGET].isna() | (df[TARGET] <= 0) | (df["Model year"] != 2025)
    # A missing predictor can be imputed inside each training-only pipeline.
    invalid |= (df[NUMERIC] <= 0).any(axis=1)
    invalid |= df[["Make", "Model", FUEL]].isna().any(axis=1)
    invalid_count = int(invalid.sum())
    df = df.loc[~invalid].reset_index(drop=True)
    # Same selected columns before/after: measure optimization, not row removal.
    comparable_before = int(df[KEEP].memory_usage(deep=True).sum())
    for col in ["Make", "Model", "Vehicle class", FUEL]:
        df[col] = df[col].astype("category")
    for col in ["Engine size (L)", "Combined (L/100 km)"]:
        df[col] = df[col].astype("float32")
    for col in ["Model year", "Cylinders", TARGET]:
        if not df[col].isna().any():
            df[col] = pd.to_numeric(df[col], downcast="unsigned")
    comparable_after = int(df[KEEP].memory_usage(deep=True).sum())
    audit = {
        "source_url": SOURCE_URL, "source_resource_id": "d589f2bc-9a85-4f65-be2f-20f17debfcb1",
        "snapshot_retrieved": "2026-11-14", "resource_label": "2025 Fuel Consumption Ratings (2026-07-24)",
        "sha256": digest, "source_encoding": "cp1252", "raw_rows": len(original),
        "raw_columns": len(original.columns), "exact_duplicates_removed": duplicates,
        "invalid_rows_removed": invalid_count, "clean_rows": len(df), "missing_values_raw": raw_missing,
        "fuel_counts": {str(k): int(v) for k, v in df[FUEL].value_counts().items()},
        "makes": int(df.Make.nunique()), "make_model_groups": int((df.Make.astype(str) + "::" + df.Model.astype(str)).nunique()),
        "memory": {"full_source_dataframe_bytes": full_memory,
                   "original_selected_columns_bytes": selected_memory,
                   "selected_before_optimization_bytes": comparable_before,
                   "selected_after_optimization_bytes": comparable_after,
                   "selected_reduction_pct": 100 * (1 - comparable_after / comparable_before)},
        "dtypes_after": {str(k): str(v) for k, v in df.dtypes.items()},
    }
    return df, audit


def numeric_pipeline(columns, estimator=None):
    # ColumnTransformer restricts features explicitly, excluding emission ratings.
    selector = ColumnTransformer([("numeric", SimpleImputer(strategy="median"), columns)], remainder="drop")
    return Pipeline([("features", selector), ("model", estimator if estimator is not None else LinearRegression())])


def candidates():
    """Predeclared simple models; no test-set tuning or hyperparameter search."""
    return {
        "Mean baseline": numeric_pipeline(NUMERIC, DummyRegressor(strategy="mean")),
        "Specifications only": numeric_pipeline(NUMERIC[:2]),
        "Fuel consumption only": numeric_pipeline([NUMERIC[2]]),
        "Phase 1 linear regression": numeric_pipeline(NUMERIC),
        "Fuel-interaction linear regression": Pipeline([
            ("features", FuelInteractionFeatures()), ("model", LinearRegression())]),
    }


def scores(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {"mae": float(mean_absolute_error(actual, predicted)), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": float(r2_score(actual, predicted))}


def cluster_bootstrap_mae(actual, predicted, groups, n_boot=2000):
    """Resample held-out catalog groups, conditional on the fixed fitted model."""
    tmp = pd.DataFrame({"group": np.asarray(groups), "ae": np.abs(np.asarray(actual)-predicted)})
    grouped = tmp.groupby("group").ae.agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    # Sample group aggregates rather than duplicating full rows 2,000 times.
    picks = rng.integers(0, len(grouped), size=(n_boot, len(grouped)))
    means = grouped["sum"].to_numpy()[picks].sum(axis=1) / grouped["count"].to_numpy()[picks].sum(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def save_figure(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)  # Free each figure before creating the next one.


def plots(df, predictions, metrics, selected_name, figures):
    """Publication figures. Every point comes from a traced source record."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#16354a", "text.color": "#16354a"})
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    upper = np.ceil(df[TARGET].max() / 25) * 25
    ax.hist(df[TARGET], bins=np.arange(75, upper + 1, 25), color="#244b70", edgecolor="white")
    ax.set(xlabel="Rated CO2 emissions (g/km)", ylabel="Vehicle records")
    save_figure(fig, figures, "emissions_distribution")

    for x, stem, xlabel in [(NUMERIC[2], "fuel_vs_co2", "Combined fuel consumption (L/100 km)"),
                             (NUMERIC[0], "engine_vs_co2", "Engine size (L)")]:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for fuel in ["X", "Z", "D", "E"]:
            sub = df.loc[df[FUEL] == fuel]
            ax.scatter(sub[x], sub[TARGET], s=18 if fuel in "XZ" else 38, alpha=.65,
                       color=COLORS[fuel], label=f"{fuel}: {FUEL_LABELS[fuel]}")
        ax.set(xlabel=xlabel, ylabel="Rated CO2 emissions (g/km)")
        ax.legend(fontsize=8, frameon=False)
        save_figure(fig, figures, stem)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bars = ax.barh([m["name"].replace(" linear regression", " LR") for m in metrics],
                  [m["test"]["mae"] for m in metrics], color=["#168a87" if m["name"]==selected_name else "#244b70" for m in metrics])
    ax.invert_yaxis(); ax.set(xlabel="Held-out MAE (g/km)")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xlim(0, max(m["test"]["mae"] for m in metrics)*1.18)
    save_figure(fig, figures, "model_mae")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, name in zip(axes, ["Phase 1 linear regression", selected_name]):
        p = predictions[name]; lo=min(p.actual.min(),p.predicted.min())-8; hi=max(p.actual.max(),p.predicted.max())+8
        ax.scatter(p.actual, p.predicted, s=16, color="#168a87", alpha=.7)
        ax.plot([lo,hi],[lo,hi], color="#7c8e98", linestyle="--", linewidth=1)
        ax.set(xlabel="Actual (g/km)", ylabel="Predicted (g/km)", xlim=(lo,hi), ylim=(lo,hi),
               title="Phase 1 numeric LR" if name.startswith("Phase 1") else "Selected fuel-interaction LR")
        ax.title.set_fontsize(9)
    fig.tight_layout()
    save_figure(fig, figures, "actual_vs_predicted")

    p = predictions[selected_name]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for fuel in ["X","Z","D","E"]:
        sub=p.loc[p[FUEL]==fuel]
        if len(sub): ax.scatter(sub.predicted,sub.residual,s=22,alpha=.7,color=COLORS[fuel],label=f"{fuel} (n={len(sub)})")
    ax.axhline(0,color="#7c8e98",linestyle="--",linewidth=1)
    ax.set(xlabel="Predicted CO2 (g/km)",ylabel="Actual - predicted (g/km)")
    ax.legend(frameon=False,fontsize=8)
    save_figure(fig, figures, "residuals")


def run(data: Path, out: Path):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    df, audit = load_and_audit(data)
    groups = df.Make.astype(str) + "::" + df.Model.astype(str)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(df, groups=groups))
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
    X = df[[*NUMERIC, FUEL]]; y = df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    group_train, group_test = groups.iloc[train], groups.iloc[test]
    folds = list(GroupKFold(n_splits=5).split(X_train, y_train, group_train))
    audit["split"] = {"method": "GroupShuffleSplit", "seed": SEED, "test_fraction_of_groups": .2,
                      "train_rows": len(train), "test_rows": len(test),
                      "train_groups": int(group_train.nunique()), "test_groups": int(group_test.nunique()),
                      "overlap_groups": 0, "group_definition": "Make + exact stripped Model string",
                      "train_fuel_counts": {str(k):int(v) for k,v in X_train[FUEL].value_counts().items()},
                      "test_fuel_counts": {str(k):int(v) for k,v in X_test[FUEL].value_counts().items()}}
    fold_records=[]; metrics=[]; models={}; predictions={}
    all_candidates=candidates()
    # Reuse identical group folds for every candidate; fit transformations anew.
    for name, pipeline in all_candidates.items():
        fold_scores=[]
        for number, (fit_idx,val_idx) in enumerate(folds, start=1):
            assert set(group_train.iloc[fit_idx]).isdisjoint(set(group_train.iloc[val_idx]))
            fitted=clone(pipeline).fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            pred=fitted.predict(X_train.iloc[val_idx]); sc=scores(y_train.iloc[val_idx],pred)
            unseen=int((~X_train.iloc[val_idx][FUEL].astype(str).isin(X_train.iloc[fit_idx][FUEL].astype(str))).sum())
            fold_records.append({"name":name,"fold":number,"fit_rows":len(fit_idx),"validation_rows":len(val_idx),"unseen_fuel_rows":unseen,**sc})
            fold_scores.append(sc)
        cv={f"{k}_mean":float(np.mean([s[k] for s in fold_scores])) for k in ["mae","mse","rmse","r2"]}
        cv.update({f"{k}_std":float(np.std([s[k] for s in fold_scores],ddof=1)) for k in ["mae","rmse","r2"]})
        models[name]=clone(pipeline).fit(X_train,y_train)
        metrics.append({"name":name,"cv":cv})
    # Select without access to held-out predictions or held-out target statistics.
    selected_name=min(metrics,key=lambda item:item["cv"]["mae_mean"])["name"]
    for item in metrics:
        name=item["name"]; predicted=models[name].predict(X_test)
        item["train"]=scores(y_train,models[name].predict(X_train))
        item["test"]=scores(y_test,predicted)
        item["test_mae_cluster_bootstrap_95ci"]=cluster_bootstrap_mae(y_test,predicted,group_test)
        p=df.iloc[test][["source_row","Make","Model",FUEL]].copy()
        p["actual"]=y_test.to_numpy(); p["predicted"]=predicted
        p["residual"]=p.actual-p.predicted; p["absolute_error"]=p.residual.abs()
        p["model"]=name; predictions[name]=p
    selected=predictions[selected_name]
    subgroup=[]
    for fuel, sub in selected.groupby(FUEL, observed=True):
        subgroup.append({"fuel":str(fuel),"count":len(sub),**scores(sub.actual,sub.predicted)})
    selected_fit=models[selected_name]
    coefficients={"intercept":float(selected_fit[-1].intercept_),
                  "coefficients":{str(k):float(v) for k,v in zip(selected_fit[0].get_feature_names_out(),selected_fit[-1].coef_)}}
    numeric_fit=models["Phase 1 linear regression"]
    coefficients["phase1_intercept"]=float(numeric_fit[-1].intercept_)
    coefficients["phase1_coefficients"]={k:float(v) for k,v in zip(NUMERIC,numeric_fit[-1].coef_)}
    corr=df[[*NUMERIC,TARGET]].corr()
    desc=df[[*NUMERIC,TARGET]].describe().T
    desc.to_csv(out/"descriptive_statistics.csv")
    corr.to_csv(out/"correlations.csv")
    df.to_csv(out/"cleaned_data.csv",index=False)
    split=df[["source_row","Make","Model"]].copy(); split["group"]=groups
    split["split"]="train"; split.loc[test,"split"]="test"; split.to_csv(out/"split_assignments.csv",index=False)
    pd.DataFrame(fold_records).to_csv(out/"cross_validation.csv",index=False)
    pd.DataFrame(subgroup).to_csv(out/"fuel_subgroup_metrics.csv",index=False)
    pd.concat(predictions.values(),ignore_index=True).to_csv(out/"test_predictions.csv",index=False)
    write_json(out/"data_audit.json",audit)
    write_json(out/"coefficients.json",coefficients)
    summary={"seed":SEED,"selection_metric":"training-only grouped 5-fold mean MAE",
             "selected_model":selected_name,"models":metrics,"subgroup_test":subgroup,
             "selected_test_median_absolute_error":float(selected.absolute_error.median()),
             "selected_test_p95_absolute_error":float(selected.absolute_error.quantile(.95)),
             "selected_test_max_absolute_error":float(selected.absolute_error.max()),
             "correlation_with_target":corr[TARGET].to_dict()}
    write_json(out/"metrics.json",summary)
    models_dir=out.parent/"models";models_dir.mkdir(exist_ok=True)
    # Persist the train-only fitted model, keeping reported test results auditable.
    joblib.dump(selected_fit,models_dir/"selected_model.joblib",compress=3)
    joblib.dump(numeric_fit,models_dir/"phase1_model.joblib",compress=3)
    metadata={"selected_model":selected_name,"fit_rows":len(train),"fuels_seen":sorted(X_train[FUEL].astype(str).unique()),
              "training_ranges":{col:[float(X_train[col].min()),float(X_train[col].max())] for col in NUMERIC},
              "dataset_sha256":audit["sha256"],"target_unit":"g/km","evaluated_scope":"2025 Canadian catalog ratings"}
    write_json(models_dir/"model_metadata.json",metadata)
    plots(df,predictions,metrics,selected_name,figures)
    environment={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                 "scikit_learn":sklearn.__version__,"matplotlib":matplotlib.__version__,"joblib":joblib.__version__,
                 "runtime_seconds_including_figures":round(time.perf_counter()-started,3)}
    write_json(out/"environment.json",environment)
    print(f"Dataset: {len(df)} records. Train/test: {len(train)}/{len(test)}. Shared groups: 0.")
    print(f"Selected by training CV: {selected_name}")
    for m in metrics:
        print(f"{m['name']}: CV MAE={m['cv']['mae_mean']:.3f}; test MAE={m['test']['mae']:.3f}; R2={m['test']['r2']:.6f}")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"data/raw/my2025-fuel-consumption-ratings.csv")
    parser.add_argument("--out",type=Path,default=ROOT/"results")
    args=parser.parse_args(); run(args.data,args.out)


if __name__ == "__main__":
    main()
