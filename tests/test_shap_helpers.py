import numpy as np
import pandas as pd

from explainability.generate_shap import _global_artifact


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
