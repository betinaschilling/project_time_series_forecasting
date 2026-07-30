import numpy as np
import pandas as pd

from explainability.generate_shap import _global_artifact, _local_artifact


class AdditiveLightGBM:
    """Emula a saída pred_contrib do LightGBM."""

    def predict(self, X, pred_contrib=False):
        contributions = np.column_stack([X["a"], 2 * X["b"]])
        base = np.full(len(X), 3.0)
        if pred_contrib:
            return np.column_stack([contributions, base])
        return base + contributions.sum(axis=1)


def test_global_shap_validates_additivity_and_ranking():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    artifact, residual = _global_artifact(AdditiveLightGBM(), "LightGBM", X)

    assert residual == 0.0
    assert artifact.iloc[0]["feature"] == "b"
    assert artifact.iloc[0]["rank"] == 1


def test_local_shap_normalizes_mixed_feature_values_for_parquet(tmp_path):
    X = pd.DataFrame({"a": [True, False], "b": [10, 20]})
    identifiers = pd.DataFrame(
        {
            "sku": [1, 2],
            "data": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "horizon": [0, 0],
        }
    )

    artifact, residual = _local_artifact(
        identifiers,
        X,
        AdditiveLightGBM(),
        "LightGBM",
        "historical",
    )
    output = tmp_path / "shap_local.parquet"
    artifact.to_parquet(output, index=False)
    restored = pd.read_parquet(output)

    assert residual == 0.0
    assert restored["feature_value"].dtype.kind == "f"
