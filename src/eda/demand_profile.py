"""Classificação de demanda por SKU com ADI e CV².

Objetivo
--------
Separar séries regulares, erráticas, intermitentes e irregulares (lumpy)
conforme os limiares de Syntetos, Boylan e Croston.

Entrada
-------
DataFrame diário com as colunas ``sku``, ``data`` e ``venda``.

Saída
-----
Uma linha por SKU com cobertura, frequência, ADI, CV² e categoria.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
CATEGORY_ORDER = [
    "Regular",
    "Errática",
    "Intermitente",
    "Irregular",
    "Sem demanda",
]


def _first_csv(path: str | Path) -> Path:
    candidate = Path(path)
    return next(candidate.glob("*.csv")) if candidate.is_dir() else candidate


def classify_demand_profiles(
    frame: pd.DataFrame,
    adi_threshold: float = ADI_THRESHOLD,
    cv2_threshold: float = CV2_THRESHOLD,
) -> pd.DataFrame:
    """Calcula ADI/CV² e classifica cada SKU.

    O ADI é o número de dias calendários observados dividido pelo número de
    dias com venda. O CV² usa somente demandas estritamente positivas, pois
    os zeros já são representados pelo intervalo entre ocorrências.
    """
    required = {"sku", "data", "venda"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    if adi_threshold <= 0 or cv2_threshold < 0:
        raise ValueError("Os limiares de ADI e CV² devem ser não negativos.")

    daily = frame[["sku", "data", "venda"]].copy()
    daily["data"] = pd.to_datetime(daily["data"], errors="coerce")
    daily["venda"] = pd.to_numeric(daily["venda"], errors="coerce")
    daily = daily.dropna(subset=["sku", "data", "venda"])
    daily = daily.groupby(["sku", "data"], as_index=False)["venda"].sum()
    if daily.empty:
        return pd.DataFrame(
            columns=[
                "sku", "primeira_data", "ultima_data", "dias_observados",
                "dias_com_venda", "percentual_dias_sem_venda", "venda_total",
                "venda_media_dia", "venda_media_quando_ocorre",
                "desvio_padrao_quando_ocorre", "adi", "cv", "cv2",
                "categoria_demanda", "ocorrencias_insuficientes",
            ]
        )

    coverage = daily.groupby("sku").agg(
        primeira_data=("data", "min"),
        ultima_data=("data", "max"),
        venda_total=("venda", "sum"),
        venda_media_dia=("venda", "mean"),
    )
    coverage["dias_observados"] = (
        coverage["ultima_data"] - coverage["primeira_data"]
    ).dt.days + 1

    positive = daily.loc[daily["venda"].gt(0)]
    occurrences = positive.groupby("sku")["venda"].agg(
        dias_com_venda="size",
        venda_media_quando_ocorre="mean",
        desvio_padrao_quando_ocorre=lambda values: values.std(ddof=0),
    )
    result = coverage.join(occurrences, how="left")
    result["dias_com_venda"] = result["dias_com_venda"].fillna(0).astype(int)
    result["percentual_dias_sem_venda"] = (
        1 - result["dias_com_venda"] / result["dias_observados"]
    )
    result["adi"] = np.where(
        result["dias_com_venda"].gt(0),
        result["dias_observados"] / result["dias_com_venda"],
        np.inf,
    )
    result["cv"] = np.where(
        result["venda_media_quando_ocorre"].gt(0),
        result["desvio_padrao_quando_ocorre"]
        / result["venda_media_quando_ocorre"],
        np.nan,
    )
    result["cv2"] = result["cv"].pow(2)
    result["ocorrencias_insuficientes"] = result["dias_com_venda"].lt(2)

    has_demand = result["dias_com_venda"].gt(0)
    frequent = result["adi"].lt(adi_threshold)
    stable = result["cv2"].lt(cv2_threshold)
    result["categoria_demanda"] = np.select(
        [
            ~has_demand,
            has_demand & frequent & stable,
            has_demand & frequent & ~stable,
            has_demand & ~frequent & stable,
            has_demand & ~frequent & ~stable,
        ],
        ["Sem demanda", "Regular", "Errática", "Intermitente", "Irregular"],
        default="Sem demanda",
    )
    result["categoria_demanda"] = pd.Categorical(
        result["categoria_demanda"],
        categories=CATEGORY_ORDER,
        ordered=True,
    )
    return result.reset_index()[
        [
            "sku", "primeira_data", "ultima_data", "dias_observados",
            "dias_com_venda", "percentual_dias_sem_venda", "venda_total",
            "venda_media_dia", "venda_media_quando_ocorre",
            "desvio_padrao_quando_ocorre", "adi", "cv", "cv2",
            "categoria_demanda", "ocorrencias_insuficientes",
        ]
    ].sort_values(["categoria_demanda", "sku"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera o perfil ADI/CV² de demanda por SKU."
    )
    parser.add_argument(
        "--input",
        default="data/processed/vendas_processed.csv",
        help="CSV processado ou diretório com part-*.csv.",
    )
    parser.add_argument(
        "--output",
        default="data/eda/sku_demand_profile.csv",
        help="Arquivo CSV de saída.",
    )
    parser.add_argument("--adi-threshold", type=float, default=ADI_THRESHOLD)
    parser.add_argument("--cv2-threshold", type=float, default=CV2_THRESHOLD)
    args = parser.parse_args()

    source = pd.read_csv(_first_csv(args.input), parse_dates=["data"])
    profiles = classify_demand_profiles(
        source,
        adi_threshold=args.adi_threshold,
        cv2_threshold=args.cv2_threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output, index=False)
    print(f"{len(profiles):,} perfis salvos em {output}")


if __name__ == "__main__":
    main()
