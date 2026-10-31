"""Optional model implementations used by the training entry points."""

from __future__ import annotations

import copy
from importlib import import_module
import time
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:
    raise ImportError(
        "Install model dependencies with "
        "`uv sync --extra models` or `pip install '.[models]'`."
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


class FTTransformerClassifier:
    """CUDA-capable FT-Transformer with sklearn-style methods.

    The input matrix must contain continuous columns first and ordinal-encoded
    categorical columns last. Unknown categories must be encoded as ``-1``.
    """

    def __init__(
        self,
        *,
        n_cat_features: int,
        n_blocks: int = 3,
        d_block: int = 192,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: float | None = None,
        batch_size: int = 256,
        max_epochs: int = 50,
        patience: int = 5,
        fixed_epochs: int | None = None,
        val_fraction: float = 0.1,
        random_state: int = 42,
        verbose: bool = True,
        log_prefix: str = "",
    ) -> None:
        self.n_cat_features = n_cat_features
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.fixed_epochs = fixed_epochs
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.log_prefix = log_prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_: nn.Module | None = None
        self.cat_cardinalities_: list[int] = []
        self.n_iter_ = 0
        self.best_epoch_ = 0
        self.loss_ = float("nan")
        self.history_: list[dict[str, float | int]] = []
        self.tuned_params_: dict[str, Any] = {}

    @staticmethod
    def _dense_float32(values: Any) -> NDArray[np.float32]:
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values, dtype=np.float32, order="C")

    def _split_columns(
        self, X: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        values = self._dense_float32(X)
        n_continuous = values.shape[1] - self.n_cat_features
        continuous = torch.from_numpy(
            np.ascontiguousarray(values[:, :n_continuous])
        )
        if self.n_cat_features == 0:
            return continuous, None

        codes = values[:, n_continuous:].astype(np.int64) + 1
        if self.cat_cardinalities_:
            upper = np.asarray(self.cat_cardinalities_, dtype=np.int64) - 1
            codes = np.minimum(codes, upper)
        return continuous, torch.from_numpy(np.ascontiguousarray(codes))

    def _build_network(self, n_continuous: int) -> nn.Module:
        try:
            transformer_class = import_module(
                "rtdl_revisiting_models"
            ).FTTransformer
        except (ImportError, AttributeError) as error:
            raise ImportError(
                "Install model dependencies with "
                "`uv sync --extra models` or `pip install '.[models]'`."
            ) from error

        kwargs = transformer_class.get_default_kwargs(n_blocks=self.n_blocks)
        kwargs.update(
            d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        )
        kwargs.pop("_is_default", None)
        return transformer_class(
            n_cont_features=n_continuous,
            cat_cardinalities=self.cat_cardinalities_,
            d_out=1,
            **kwargs,
        ).to(self.device)

    def _logits(
        self,
        network: nn.Module,
        continuous: torch.Tensor,
        categorical: torch.Tensor | None,
    ) -> NDArray[np.float32]:
        use_amp = self.device.type == "cuda"
        network.eval()
        outputs: list[NDArray[np.float32]] = []
        with torch.no_grad():
            for start in range(0, len(continuous), self.batch_size):
                stop = start + self.batch_size
                x_continuous = continuous[start:stop].to(self.device)
                x_categorical = (
                    None
                    if categorical is None
                    else categorical[start:stop].to(self.device)
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = network(x_continuous, x_categorical)
                outputs.append(logits.float().cpu().numpy().reshape(-1))
        return np.concatenate(outputs)

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
    ) -> Self:
        """Train with weighted BCE and optional validation-AP stopping."""

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        targets = np.asarray(y, dtype=np.float32).reshape(-1)
        values = self._dense_float32(X)
        if X_val is None and self.fixed_epochs is None:
            train_rows, validation_rows = train_test_split(
                np.arange(len(targets)),
                test_size=self.val_fraction,
                stratify=targets,
                random_state=self.random_state,
            )
            X_val, y_val = values[validation_rows], targets[validation_rows]
            values, targets = values[train_rows], targets[train_rows]

        n_continuous = values.shape[1] - self.n_cat_features
        if self.n_cat_features:
            train_codes = values[:, n_continuous:].astype(np.int64) + 1
            self.cat_cardinalities_ = [
                int(column.max()) + 1 for column in train_codes.T
            ]
        continuous, categorical = self._split_columns(values)
        labels = torch.from_numpy(targets.reshape(-1, 1))

        use_validation = X_val is not None
        if use_validation:
            val_continuous, val_categorical = self._split_columns(X_val)
            val_labels = np.asarray(y_val, dtype=np.float32).reshape(-1)

        network = self._build_network(n_continuous)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        positive_weight = (
            None
            if self.pos_weight is None
            else torch.tensor(
                [self.pos_weight], dtype=torch.float32, device=self.device
            )
        )
        loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        generator = torch.Generator().manual_seed(self.random_state)

        epochs = (
            self.max_epochs
            if use_validation or self.fixed_epochs is None
            else self.fixed_epochs
        )
        best_score = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        self.history_ = []

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            network.train()
            epoch_loss = 0.0
            order = torch.randperm(len(continuous), generator=generator)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                x_continuous = continuous[batch].to(self.device)
                x_categorical = (
                    None
                    if categorical is None
                    else categorical[batch].to(self.device)
                )
                batch_labels = labels[batch].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = network(x_continuous, x_categorical)
                    loss = loss_function(logits.float(), batch_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.detach().item() * len(batch)

            epoch_loss /= len(order)
            self.n_iter_ = epoch + 1
            self.loss_ = epoch_loss
            record: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
            }

            if use_validation:
                validation_logits = self._logits(
                    network, val_continuous, val_categorical
                )
                score = (
                    float(average_precision_score(val_labels, validation_logits))
                    if val_labels.sum() > 0
                    else 0.0
                )
                record["validation_average_precision"] = score
                if score > best_score:
                    best_score = score
                    stale_epochs = 0
                    self.best_epoch_ = epoch + 1
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in network.state_dict().items()
                    }
                else:
                    stale_epochs += 1

            record["seconds"] = time.perf_counter() - epoch_start
            self.history_.append(record)
            if self.verbose:
                prefix = f"{self.log_prefix} | " if self.log_prefix else ""
                validation_text = (
                    f" | validation AP {score:.4f} "
                    f"(best {best_score:.4f} @ {self.best_epoch_})"
                    if use_validation
                    else ""
                )
                print(
                    f"{prefix}epoch {epoch + 1}/{epochs} | "
                    f"loss {epoch_loss:.5f}{validation_text} | "
                    f"{record['seconds']:.1f} s/epoch",
                    flush=True,
                )
            if use_validation and stale_epochs >= self.patience:
                break

        if best_state is not None:
            network.load_state_dict(best_state)
        else:
            self.best_epoch_ = self.n_iter_
        self.network_ = network
        return self

    def fit_with_validation(
        self, X: Any, y: Any, X_val: Any, y_val: Any
    ) -> Self:
        return self.fit(X, y, X_val, y_val)

    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return two-column probabilities for binary evaluation."""

        if self.network_ is None:
            raise RuntimeError("Call fit before predict_proba.")
        continuous, categorical = self._split_columns(X)
        logits = self._logits(self.network_, continuous, categorical)
        positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        return np.column_stack([1.0 - positive, positive]).astype(
            np.float32, copy=False
        )
