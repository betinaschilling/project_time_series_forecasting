import numpy as np
import pandas as pd

from eda.demand_profile import classify_demand_profiles


def series(sku, values):
    return pd.DataFrame(
        {
            "sku": sku,
            "data": pd.date_range("2025-01-01", periods=len(values)),
            "venda": values,
        }
    )


def test_classifies_regular_demand():
    result = classify_demand_profiles(series("regular", [10, 11, 10, 11]))
    row = result.iloc[0]
    assert row["adi"] == 1
    assert row["cv2"] < 0.49
    assert row["categoria_demanda"] == "Regular"


def test_classifies_erratic_demand():
    result = classify_demand_profiles(series("erratic", [1, 100, 1, 100]))
    assert result.iloc[0]["categoria_demanda"] == "Errática"


def test_classifies_intermittent_demand():
    values = [10, 0, 0, 10, 0, 0, 10, 0]
    result = classify_demand_profiles(series("intermittent", values))
    assert result.iloc[0]["categoria_demanda"] == "Intermitente"


def test_classifies_irregular_demand():
    values = [1, 0, 0, 100, 0, 0, 1, 0]
    result = classify_demand_profiles(series("irregular", values))
    assert result.iloc[0]["categoria_demanda"] == "Irregular"


def test_adi_uses_calendar_coverage_and_cv2_ignores_zeroes():
    result = classify_demand_profiles(series("sku", [10, 0, 10, 0]))
    row = result.iloc[0]
    assert row["adi"] == 2
    assert row["cv2"] == 0
    assert row["percentual_dias_sem_venda"] == 0.5


def test_marks_sku_without_positive_demand():
    result = classify_demand_profiles(series("zero", [0, 0, 0]))
    row = result.iloc[0]
    assert np.isinf(row["adi"])
    assert row["categoria_demanda"] == "Sem demanda"
