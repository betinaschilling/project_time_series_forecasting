import numpy as np
import pandas as pd

from features.forecasting_features import generate_all_features, recursive_forecast


class LagWeekdayModel:
    """Modelo mínimo para tornar a previsão temporalmente verificável."""

    def predict(self, X):
        return X["lag_1"].to_numpy() + X["weekday"].to_numpy()


def history():
    return pd.DataFrame(
        {
            "sku": ["A"] * 7,
            "data": pd.date_range("2025-01-01", periods=7, freq="D"),
            "venda": np.arange(10.0, 17.0),
            "is_imputed": False,
        }
    )


def test_features_do_not_use_current_target():
    frame = history()
    featured = generate_all_features(frame)
    assert featured.iloc[-1]["lag_1"] == frame.iloc[-2]["venda"]
    assert featured.iloc[-1]["roll_mean_7"] == frame.iloc[:-1]["venda"].mean()


def test_recursive_forecast_builds_next_date_before_prediction():
    forecast, snapshots = recursive_forecast(
        model=LagWeekdayModel(),
        history=history(),
        feature_columns=["lag_1", "weekday"],
        horizon=2,
    )

    first_date = pd.Timestamp("2025-01-08")
    assert forecast.iloc[0]["data"] == first_date
    assert snapshots.iloc[0]["weekday"] == first_date.weekday()
    assert snapshots.iloc[0]["lag_1"] == 16.0
    assert forecast.iloc[0]["prediction"] == 16.0 + first_date.weekday()

    second_date = pd.Timestamp("2025-01-09")
    assert snapshots.iloc[1]["lag_1"] == forecast.iloc[0]["prediction"]
    assert snapshots.iloc[1]["weekday"] == second_date.weekday()


def test_recursive_forecast_discards_history_outside_required_lookback():
    long_history = pd.concat(
        [history()] * 10,
        ignore_index=True,
    )
    long_history["data"] = pd.date_range("2024-01-01", periods=len(long_history))
    forecast, _ = recursive_forecast(
        model=LagWeekdayModel(),
        history=long_history,
        feature_columns=["lag_1", "weekday"],
        horizon=1,
    )
    assert len(forecast) == 1
