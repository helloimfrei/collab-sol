from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from collab_sol.models import TorchMLPClassifier


def test_torch_mlp_fits_and_returns_binary_probabilities() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(64, 4)).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    model = TorchMLPClassifier(
        hidden_width=8,
        n_layers=1,
        dropout=0.0,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=16,
        max_epochs=2,
        patience=2,
        random_state=42,
    )

    assert model.fit(features, labels) is model
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (64, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(64))
