#!/usr/bin/env python
"""Gera atribuições SHAP globais e locais para LightGBM e CatBoost.

As implementações nativas dos dois modelos são usadas para evitar uma
dependência adicional no ambiente do Streamlit:

- LightGBM: ``predict(..., pred_contrib=True)``
- CatBoost: ``get_feature_importance(..., type="ShapValues")``

Em ambos os casos, a última coluna é o valor-base da explicação.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features.forecasting_features import (
    generate_all_features,
    recursive_forecast,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    artifact: Path


def _first_csv(path: str | Path) -> Path:
    candidate = Path(path)
    return next(candidate.glob("*.csv")) if candidate.is_dir() else candidate


def _load_metadata(artifact: Path) -> dict:
    metadata_path = Path(f"{artifact}.metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_model(spec: ModelSpec):
    if spec.name == "LightGBM":
        return joblib.load(spec.artifact)
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(spec.artifact))
    return model


def _model_features(model, spec: ModelSpec) -> list[str]:
    metadata = _load_metadata(spec.artifact)
    if metadata.get("feature_columns"):
        return list(metadata["feature_columns"])
    if spec.name == "LightGBM":
        return list(model.feature_name_)
    return list(model.feature_names_)


def _contributions(model, model_name: str, X: pd.DataFrame):
    if model_name == "LightGBM":
        matrix = np.asarray(model.predict(X, pred_contrib=True), dtype=float)
    else:
        from catboost import Pool

        matrix = np.asarray(
            model.get_feature_importance(Pool(X), type="ShapValues"),
            dtype=float,
        )
    shap_values = matrix[:, :-1]
    base_values = matrix[:, -1]
    predictions = base_values + shap_values.sum(axis=1)
    return shap_values, base_values, predictions


def _recent_sample(
    frame: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Prioriza o regime recente e limita custo/memória de explicação."""
    ordered = frame.sort_values("data")
    if len(ordered) <= max_rows:
        return ordered
    recent_pool = ordered.tail(max_rows * 3)
    return (
        recent_pool.sample(max_rows, random_state=random_state)
        .sort_values(["data", "sku"])
        .reset_index(drop=True)
    )


def _global_artifact(
    model,
    model_name: str,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    shap_values, base_values, reconstructed = _contributions(model, model_name, X)
    direct = np.asarray(model.predict(X), dtype=float)
    residual = float(np.max(np.abs(reconstructed - direct)))

    output = pd.DataFrame(
        {
            "model": model_name,
            "feature": X.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "positive_share": (shap_values > 0).mean(axis=0),
            "sample_rows": len(X),
            "mean_base_value": float(np.mean(base_values)),
        }
    )
    output = output.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1)
    return output, residual


def _local_artifact(
    identifiers: pd.DataFrame,
    X: pd.DataFrame,
    model,
    model_name: str,
    scope: str,
) -> tuple[pd.DataFrame, float]:
    shap_values, base_values, reconstructed = _contributions(model, model_name, X)
    direct = np.asarray(model.predict(X), dtype=float)
    residual = float(np.max(np.abs(reconstructed - direct)))

    row_count, feature_count = shap_values.shape
    repeated_ids = identifiers.reset_index(drop=True).loc[
        identifiers.reset_index(drop=True).index.repeat(feature_count)
    ].reset_index(drop=True)
    repeated_ids["model"] = model_name
    repeated_ids["scope"] = scope
    repeated_ids["feature"] = np.tile(np.asarray(X.columns), row_count)
    repeated_ids["feature_value"] = X.to_numpy().reshape(-1)
    repeated_ids["shap_value"] = shap_values.reshape(-1)
    repeated_ids["abs_shap"] = np.abs(repeated_ids["shap_value"])
    repeated_ids["base_value"] = np.repeat(base_values, feature_count)
    repeated_ids["prediction"] = np.repeat(direct, feature_count)
    return repeated_ids, residual


def generate_explanations(
    features_csv: str,
    lgbm_model: str,
    catboost_model: str,
    output_dir: str,
    horizon: int = 7,
    max_global_rows: int = 5000,
    max_local_rows: int = 5000,
    max_future_skus: int = 500,
    local_days: int = 30,
    random_state: int = 42,
) -> dict:
    raw = pd.read_csv(_first_csv(features_csv), parse_dates=["data"])
    featured = generate_all_features(raw)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    specs = (
        ModelSpec("LightGBM", Path(lgbm_model)),
        ModelSpec("CatBoost", Path(catboost_model)),
    )
    global_parts: list[pd.DataFrame] = []
    local_parts: list[pd.DataFrame] = []
    validation: list[dict] = []

    for spec in specs:
        model = _load_model(spec)
        feature_columns = _model_features(model, spec)
        missing = set(feature_columns).difference(featured.columns)
        if missing:
            raise ValueError(
                f"{spec.name}: features ausentes na base: {sorted(missing)}"
            )

        global_rows = _recent_sample(featured, max_global_rows, random_state)
        global_frame, global_residual = _global_artifact(
            model,
            spec.name,
            global_rows[feature_columns],
        )
        global_parts.append(global_frame)
        validation.append(
            {
                "model": spec.name,
                "scope": "global",
                "max_additivity_residual": global_residual,
                "rows": len(global_rows),
            }
        )

        cutoff = featured["data"].max() - pd.Timedelta(days=local_days - 1)
        historical = featured.loc[featured["data"].ge(cutoff)].copy()
        historical = _recent_sample(historical, max_local_rows, random_state)
        historical_ids = historical[["sku", "data"]].copy()
        historical_ids["horizon"] = 0
        local_history, history_residual = _local_artifact(
            historical_ids,
            historical[feature_columns],
            model,
            spec.name,
            "historical",
        )
        local_parts.append(local_history)
        validation.append(
            {
                "model": spec.name,
                "scope": "historical",
                "max_additivity_residual": history_residual,
                "rows": len(historical),
            }
        )

        recent_cutoff = featured["data"].max() - pd.Timedelta(days=29)
        active_skus = (
            featured.loc[featured["data"].ge(recent_cutoff)]
            .groupby("sku")["venda"]
            .sum()
            .sort_values(ascending=False)
            .head(max_future_skus)
            .index
        )
        future_history = featured.loc[featured["sku"].isin(active_skus)]
        _, future_features = recursive_forecast(
            model=model,
            history=future_history,
            feature_columns=feature_columns,
            horizon=horizon,
        )
        future_ids = future_features[["sku", "data", "horizon"]]
        local_future, future_residual = _local_artifact(
            future_ids,
            future_features[feature_columns],
            model,
            spec.name,
            "forecast",
        )
        local_parts.append(local_future)
        validation.append(
            {
                "model": spec.name,
                "scope": "forecast",
                "max_additivity_residual": future_residual,
                "rows": len(future_features),
            }
        )

    global_output = pd.concat(global_parts, ignore_index=True)
    local_output = pd.concat(local_parts, ignore_index=True)
    validation_output = pd.DataFrame(validation)

    global_output.to_csv(output_path / "shap_global.csv", index=False)
    try:
        local_output.to_parquet(output_path / "shap_local.parquet", index=False)
        local_file = "shap_local.parquet"
    except (ImportError, ModuleNotFoundError):
        local_output.to_csv(output_path / "shap_local.csv", index=False)
        local_file = "shap_local.csv"
    validation_output.to_csv(output_path / "shap_validation.csv", index=False)

    manifest = {
        "global_file": "shap_global.csv",
        "local_file": local_file,
        "validation_file": "shap_validation.csv",
        "horizon": horizon,
        "future_sku_coverage": min(max_future_skus, featured["sku"].nunique()),
        "max_additivity_residual": float(
            validation_output["max_additivity_residual"].max()
        ),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="data/features/vendas_features.csv")
    parser.add_argument("--lgbm-model", default="data/models/lgbm_model.pkl")
    parser.add_argument("--catboost-model", default="data/models/catboost_model.cbm")
    parser.add_argument("--output-dir", default="data/explainability")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--max-global-rows", type=int, default=5000)
    parser.add_argument("--max-local-rows", type=int, default=5000)
    parser.add_argument("--max-future-skus", type=int, default=500)
    parser.add_argument("--local-days", type=int, default=30)
    args = parser.parse_args()

    manifest = generate_explanations(
        features_csv=args.features_csv,
        lgbm_model=args.lgbm_model,
        catboost_model=args.catboost_model,
        output_dir=args.output_dir,
        horizon=args.horizon,
        max_global_rows=args.max_global_rows,
        max_local_rows=args.max_local_rows,
        max_future_skus=args.max_future_skus,
        local_days=args.local_days,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
