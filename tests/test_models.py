from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

pytest.importorskip("torch")

from collab_sol.models import FTTransformerClassifier, TorchMLPClassifier


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


def test_ft_transformer_fits_and_returns_binary_probabilities() -> None:
    if find_spec("rtdl_revisiting_models") is None:
        pytest.skip("FT-Transformer optional dependency is not installed.")

    rng = np.random.default_rng(42)
    continuous = rng.normal(size=(64, 2)).astype(np.float32)
    categories = rng.integers(0, 3, size=(64, 1)).astype(np.float32)
    features = np.column_stack([continuous, categories])
    labels = (continuous[:, 0] > 0).astype(int)
    model = FTTransformerClassifier(
        n_cat_features=1,
        n_blocks=1,
        d_block=8,
        attention_n_heads=2,
        batch_size=16,
        fixed_epochs=1,
        max_epochs=1,
        verbose=False,
    )

    assert model.fit(features, labels) is model
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (64, 2)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(64))
