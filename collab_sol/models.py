"""Optional model implementations used by the training entry points."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:
    raise ImportError(
        "Install the PyTorch model dependencies with "
        "`uv sync --extra gpu` or `pip install '.[gpu]'`."
    ) from error


class TorchMLPClassifier:
    """CUDA-capable binary MLP with sklearn-style fit and predict_proba."""

    def __init__(
        self,
        *,
        hidden_width: int,
        n_layers: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int = 50,
        patience: int = 5,
        tolerance: float = 5e-4,
        random_state: int = 42,
    ) -> None:
        self.hidden_width = hidden_width
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_: nn.Sequential | None = None
        self.n_iter_ = 0
        self.loss_ = float("nan")

    @staticmethod
    def _dense_float32(values: Any) -> NDArray[np.float32]:
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values, dtype=np.float32, order="C")

    def _build_network(self, input_width: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layer_width = input_width
        for _ in range(self.n_layers):
            layers.extend(
                [
                    nn.Linear(layer_width, self.hidden_width),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            layer_width = self.hidden_width
        layers.append(nn.Linear(layer_width, 1))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X: Any, y: Any) -> Self:
        """Fit the network using BCE loss and train-loss early stopping."""

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        features = torch.from_numpy(self._dense_float32(X))
        targets = torch.from_numpy(
            np.asarray(y, dtype=np.float32).reshape(-1, 1)
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(features, targets),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

        network = self._build_network(features.shape[1])
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_function = nn.BCEWithLogitsLoss()
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        for epoch in range(self.max_epochs):
            network.train()
            epoch_loss = 0.0
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(
                    self.device, non_blocking=use_amp
                )
                batch_targets = batch_targets.to(
                    self.device, non_blocking=use_amp
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = network(batch_features)
                    loss = loss_function(logits, batch_targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.detach().item() * len(batch_features)

            epoch_loss /= len(features)
            self.n_iter_ = epoch + 1
            if best_loss - epoch_loss > self.tolerance:
                best_loss = epoch_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("The MLP did not complete a training epoch.")
        network.load_state_dict(best_state)
        self.network_ = network
        self.loss_ = best_loss
        print(
            f"PyTorch MLP | device={self.device} | "
            f"epochs={self.n_iter_} | loss={self.loss_:.6f}"
        )
        return self

    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return two-column probabilities for the negative and positive classes."""

        if self.network_ is None:
            raise RuntimeError("Call fit before predict_proba.")

        features = torch.from_numpy(self._dense_float32(X))
        loader = DataLoader(
            TensorDataset(features),
            batch_size=self.batch_size * 2,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        probabilities: list[NDArray[np.float32]] = []
        self.network_.eval()
        with torch.no_grad():
            for (batch_features,) in loader:
                batch_features = batch_features.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    logits = self.network_(batch_features)
                probabilities.append(
                    torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
                )

        positive = np.concatenate(probabilities)
        return np.column_stack([1.0 - positive, positive]).astype(
            np.float32, copy=False
        )
