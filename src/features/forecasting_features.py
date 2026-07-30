"""Feature engineering compartilhada pelos modelos de forecasting.

O módulo mantém a mesma definição de atributos no treino, na previsão
recursiva e na explicabilidade. A linha referente ao próximo dia é criada
antes do cálculo dos atributos para impedir deslocamento temporal de
calendário, lags e médias móveis.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 7, 14)
DEFAULT_WINDOWS = (7, 14, 30)
ID_COLUMNS = ("sku", "data")
TARGET_COLUMN = "venda"


def generate_all_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Gera atributos temporais sem utilizar o target da própria linha."""
    required = {"sku", "data", TARGET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    df_feat = df.copy()
    df_feat["data"] = pd.to_datetime(df_feat["data"])
    df_feat = df_feat.sort_values(["sku", "data"]).reset_index(drop=True)

    if "is_imputed" not in df_feat:
        df_feat["is_imputed"] = False
    df_feat["is_imputed"] = df_feat["is_imputed"].fillna(False).astype(bool)
    df_feat["was_imputed"] = df_feat["is_imputed"].astype(int)

    grouped = df_feat.groupby("sku", sort=False)[TARGET_COLUMN]
    for lag in lags:
        df_feat[f"lag_{lag}"] = grouped.shift(lag).fillna(0.0)

    for window in windows:
        df_feat[f"roll_mean_{window}"] = grouped.transform(
            lambda values: values.shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        ).fillna(0.0)

    df_feat["weekday"] = df_feat["data"].dt.weekday
    df_feat["is_weekend"] = df_feat["weekday"].isin([5, 6]).astype(int)
    df_feat["month"] = df_feat["data"].dt.month
    return df_feat


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna atributos na ordem do DataFrame, excluindo IDs e target."""
    excluded = {*ID_COLUMNS, TARGET_COLUMN}
    return [column for column in df.columns if column not in excluded]


def _next_rows(history: pd.DataFrame, next_date: pd.Timestamp) -> pd.DataFrame:
    """Cria uma linha ainda sem target para cada SKU no próximo dia."""
    skus = pd.Index(history["sku"].drop_duplicates(), name="sku")
    next_rows = pd.DataFrame({"sku": skus, "data": next_date})
    next_rows[TARGET_COLUMN] = np.nan
    next_rows["is_imputed"] = False
    return next_rows


def recursive_forecast(
    model,
    history: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    predictor: Callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prevê recursivamente e devolve previsões e atributos de cada passo.

    A previsão do horizonte h é calculada a partir da linha datada em h.
    O valor previsto só é inserido no histórico depois da criação e uso
    dos atributos daquele horizonte.
    """
    if horizon < 1:
        raise ValueError("horizon deve ser maior ou igual a 1")

    predict = predictor or (lambda fitted_model, frame: fitted_model.predict(frame))
    history_columns = [
        column
        for column in ("sku", "data", TARGET_COLUMN, "is_imputed")
        if column in history.columns
    ]
    work = history[history_columns].copy()
    work["data"] = pd.to_datetime(work["data"])
    # Apenas os 30 últimos registros por SKU são necessários para os lags e
    # janelas atualmente definidos. O recorte evita recomputar todo o histórico
    # a cada horizonte, sobretudo em bases com milhares de SKUs.
    lookback = max((*DEFAULT_LAGS, *DEFAULT_WINDOWS))
    work = (
        work.sort_values(["sku", "data"])
        .groupby("sku", group_keys=False, sort=False)
        .tail(lookback)
        .reset_index(drop=True)
    )

    prediction_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []

    for step in range(1, horizon + 1):
        next_date = work["data"].max() + pd.Timedelta(days=1)
        candidates = _next_rows(work, next_date)
        expanded = pd.concat([work, candidates], ignore_index=True, sort=False)
        featured = generate_all_features(expanded)
        future = featured.loc[featured["data"].eq(next_date)].copy()

        missing = set(feature_columns).difference(future.columns)
        if missing:
            raise ValueError(
                f"Features requeridas pelo modelo não foram geradas: {sorted(missing)}"
            )

        X_future = future[feature_columns]
        prediction = np.asarray(predict(model, X_future), dtype=float)

        forecast = future[["sku", "data"]].copy()
        forecast["prediction"] = prediction
        forecast["horizon"] = step
        prediction_parts.append(forecast)

        snapshot = future[["sku", "data", *feature_columns]].copy()
        snapshot["horizon"] = step
        feature_parts.append(snapshot)

        observed_next = candidates.copy()
        observed_next[TARGET_COLUMN] = prediction
        work = pd.concat([work, observed_next], ignore_index=True, sort=False)

    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(feature_parts, ignore_index=True),
    )
