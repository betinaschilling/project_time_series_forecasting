#!/usr/bin/env python
"""Treinamento e forecast recursivo com CatBoost."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from features.forecasting_features import (
    generate_all_features,
    infer_feature_columns,
    recursive_forecast,
)


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    denominator = y_true.sum()
    return {
        "rmse": math.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": (abs((y_true - y_pred) / y_true.replace(0, 1)).mean()) * 100,
        "wmape": (
            abs(y_true - y_pred).sum() / denominator * 100
            if denominator
            else float("nan")
        ),
    }


class CBTrainer:
    def __init__(
        self,
        features_csv,
        model_out,
        forecast_out,
        log_path,
        folds=5,
        horizon=7,
        metadata_out=None,
    ):
        self.features_csv = features_csv
        self.model_out = Path(model_out)
        self.forecast_out = Path(forecast_out)
        self.metadata_out = Path(metadata_out or f"{model_out}.metadata.json")
        self.log_path = log_path
        self.folds = folds
        self.horizon = horizon
        self.feature_cols: list[str] = []
        self._setup_logging()

    def _setup_logging(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def load_data(self) -> pd.DataFrame:
        path = Path(self.features_csv)
        csv = next(path.glob("*.csv")) if path.is_dir() else path
        raw = pd.read_csv(csv, parse_dates=["data"])
        frame = generate_all_features(raw)
        self.feature_cols = infer_feature_columns(frame)
        logging.getLogger("pipeline").info("Features CatBoost: %s", self.feature_cols)
        return frame

    def cross_validate(self, df: pd.DataFrame):
        X, y = df[self.feature_cols], df["venda"]
        rmses = []
        for fold, (train, validation) in enumerate(
            TimeSeriesSplit(n_splits=self.folds).split(X), 1
        ):
            model = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                verbose=False,
                random_state=42,
            )
            model.fit(X.iloc[train], y.iloc[train])
            metrics = evaluate(y.iloc[validation], model.predict(X.iloc[validation]))
            rmses.append(metrics["rmse"])
            logging.getLogger("pipeline").info(
                "CatBoost fold %s RMSE %.4f", fold, metrics["rmse"]
            )
        return rmses

    def train_and_save(self, df: pd.DataFrame):
        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            verbose=False,
            random_state=42,
        )
        model.fit(df[self.feature_cols], df["venda"])
        self.model_out.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(self.model_out))
        self.metadata_out.write_text(
            json.dumps(
                {
                    "model": "CatBoost",
                    "feature_columns": self.feature_cols,
                    "target": "venda",
                    "horizon": self.horizon,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return model

    def rolling_forecast(self, model, df: pd.DataFrame) -> pd.DataFrame:
        forecast, _ = recursive_forecast(
            model=model,
            history=df,
            feature_columns=self.feature_cols,
            horizon=self.horizon,
        )
        return forecast.rename(columns={"prediction": "cb_pred"})[
            ["sku", "data", "cb_pred"]
        ]

    def run(self):
        logger = logging.getLogger("pipeline")
        logger.info("==== INÍCIO CATBOOST ====")
        df = self.load_data()
        self.cross_validate(df)
        model = self.train_and_save(df)

        fitted = df[["sku", "data"]].copy()
        fitted["cb_pred"] = model.predict(df[self.feature_cols])
        future = self.rolling_forecast(model, df)
        output = pd.concat([fitted, future], ignore_index=True).sort_values(
            ["sku", "data"]
        )
        self.forecast_out.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(self.forecast_out, index=False)
        logger.info("Forecast CatBoost salvo em %s", self.forecast_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="data/features/vendas_features.csv")
    parser.add_argument("--model-out", default="data/models/catboost_model.cbm")
    parser.add_argument("--forecast-out", default="data/models/catboost_forecast.csv")
    parser.add_argument("--metadata-out", default=None)
    parser.add_argument("--log", default="data/logs/pipeline.log")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()

    CBTrainer(
        features_csv=args.features_csv,
        model_out=args.model_out,
        forecast_out=args.forecast_out,
        metadata_out=args.metadata_out,
        log_path=args.log,
        folds=args.folds,
        horizon=args.horizon,
    ).run()


if __name__ == "__main__":
    main()
