"""Neural operator building blocks for Saint-Venant equation surrogates.

This module defines the normalization helpers, Fourier feature embedding, a
single neural operator, and a diameter-indexed multi-expert wrapper used for
inference from saved checkpoints.
"""

__author__ = "Ruozhou Lin <ruozhou.lin@my.cityu.edu.hk>"

import math
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch import nn


class ScalerStandard:
    """Feature-wise standard scaler backed by torch tensors."""

    statistic_keys = ("data_min_", "data_max_", "mean_", "var_", "sigma_", "scale_")

    def __init__(self, X: Tensor, device: str | torch.device | None = None) -> None:
        """Fit scaler statistics from ``X``.

        Args:
            X: Training samples with features along dimension 1.
            device: Optional device where scaler tensors should be stored.
        """
        self.data_min_: Tensor = X.detach().min(dim=0, keepdim=True).values
        self.data_max_: Tensor = X.detach().max(dim=0, keepdim=True).values
        self.mean_: Tensor = torch.mean(X.detach(), dim=0, keepdim=True)
        self.var_: Tensor = torch.var(X.detach(), correction=0, dim=0, keepdim=True)
        self.sigma_: Tensor = torch.sqrt(self.var_)
        self.scale_: Tensor = self.sigma_
        # Keep constant-valued features finite during standardization.
        self.scale_[self.scale_ == 0] = 1
        if device is not None:
            self.to(device)

    @classmethod
    def load(
        cls,
        data: dict[str, Tensor],
        device: str | torch.device | None = None,
    ) -> "ScalerStandard":
        """Restore scaler statistics without refitting from raw samples.

        Args:
            data: Mapping containing all keys in ``statistic_keys``.
            device: Optional target device for the restored tensors.
        """
        scaler = cls.__new__(cls)
        for key in cls.statistic_keys:
            setattr(scaler, key, torch.as_tensor(data[key]).detach().clone())
        if device is not None:
            scaler.to(device)
        return scaler

    def transform(self, X: Tensor) -> Tensor:
        """Standardize samples using fitted mean and scale."""
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X_std: Tensor) -> Tensor:
        """Map standardized samples back to the original feature scale."""
        return X_std * self.scale_ + self.mean_

    def to(self, device: str | torch.device) -> None:
        """Move all stored scaler tensors to ``device``."""
        self.data_min_ = self.data_min_.to(device)
        self.data_max_ = self.data_max_.to(device)
        self.mean_ = self.mean_.to(device)
        self.var_ = self.var_.to(device)
        self.sigma_ = self.sigma_.to(device)
        self.scale_ = self.scale_.to(device)


class FourierEmbedding(nn.Module):
    """Random Fourier feature embedding for continuous input coordinates."""

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int,
        scale: float = 10.0,
        trainable: bool = True,
    ):
        """Fourier feature mapping.

        Args:
            input_dim (int): input dimensionality
            num_frequencies (int): number of frequency channels
            scale (float): initial scale of the random frequencies
            trainable (bool): if True the frequency matrix B is trainable; if False it is fixed
        """
        super().__init__()
        # By default make B trainable. Set trainable=False to freeze (previous behavior).
        self.B = nn.Parameter(
            torch.randn(input_dim, num_frequencies) * scale,
            requires_grad=trainable,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project inputs to concatenated sine and cosine features."""
        # x: (batch, input_dim); output: (batch, 2 * num_frequencies)
        x_proj = 2 * math.pi * x @ self.B  # (batch, num_frequencies)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SaintVenantNeuralOperator(nn.Module):
    """Fourier-embedded MLP surrogate for Saint-Venant operator prediction."""

    def __init__(
        self,
        layers: list[int],
        activation_function=nn.ReLU,
        dropout_rates: list[float] | None = None,
        isBatchNormalizationActivated: bool = False,
        isLayerNormalizationActivated: bool = False,
        isInputNormalizationActivated: bool = False,
        num_frequencies: int = 64,
        scale: float = 10.0,
    ):
        """Build a Fourier feature encoder followed by a configurable MLP.

        Args:
            layers: Network layer sizes, including raw input size first and
                output size last. The first hidden input is replaced by the
                Fourier embedding width.
            activation_function: ``nn.Module`` class used after hidden layers.
            dropout_rates: Optional per-hidden-layer dropout probabilities.
            isBatchNormalizationActivated: Whether to apply batch norm after
                hidden linear layers.
            isLayerNormalizationActivated: Whether to apply layer norm after
                hidden linear layers.
            isInputNormalizationActivated: Whether to batch-normalize embedded
                Fourier features before the first linear layer.
            num_frequencies: Number of Fourier frequency channels.
            scale: Initial random frequency scale for the Fourier embedding.
        """
        super().__init__()
        # Scalers are loaded separately from model weights and used by predict().
        self.scaler_input: ScalerStandard | None = None
        self.scaler_output: ScalerStandard | None = None

        input_dim = layers[0]
        # FourierEmbedding expands each frequency into sin and cos channels.
        layers = [2 * num_frequencies, *layers[1:]]
        self.embed = FourierEmbedding(input_dim, num_frequencies, scale)
        self.net = nn.Sequential()

        if isInputNormalizationActivated:
            self.net.add_module("BatchNorm_input", nn.BatchNorm1d(layers[0]))
        for i in range(len(layers) - 2):
            self.net.add_module(f"Layer_{i}", nn.Linear(layers[i], layers[i + 1]))
            if isBatchNormalizationActivated:
                self.net.add_module(f"    BatchNorm_{i}", nn.BatchNorm1d(layers[i + 1]))
            if isLayerNormalizationActivated:
                self.net.add_module(f"    LayerNorm_{i}", nn.LayerNorm(layers[i + 1]))
            self.net.add_module(f"  Activation_{i}", activation_function())
            if dropout_rates is not None and dropout_rates[i] > 0:
                self.net.add_module(f"  Dropout_{i}", nn.Dropout(p=dropout_rates[i]))
        self.net.add_module("Output", nn.Linear(layers[-2], layers[-1]))

    def forward(self, x: Tensor) -> Tensor:
        """Run normalized model inputs through embedding and MLP head."""
        x = self.embed(x)
        return self.net(x)

    def predict(self, features: Tensor) -> Tensor:
        """Predict on physical-scale features and return physical-scale outputs."""
        if self.scaler_input is None or self.scaler_output is None:
            raise RuntimeError("Load or set scalers before calling predict().")

        self.eval()
        with torch.no_grad():
            x = self.scaler_input.transform(features)
            y = self(x)
            return self.scaler_output.inverse_transform(y)


class MultiExpertNeuralOperator:
    """The proposed multi-expert neural operator for sewer hydraulics."""

    # Diameters correspond to the expert keys expected in the checkpoint.
    diameters: list[int] = [
        150,
        225,
        300,
        375,
        450,
        600,
        750,
        900,
        1050,
        1200,
        1350,
        1500,
        1650,
        1800,
    ]

    def __init__(self):
        """Create an empty multi-expert operator registry."""
        self.model_dict: dict[int, SaintVenantNeuralOperator] = {}
        self.is_model_loaded = False

    def get_expert(self, diameter: int) -> SaintVenantNeuralOperator:
        """Get the expert model corresponding to a given diameter."""
        self.check_nominal_diameter(diameter)
        if not self.is_model_loaded:
            raise RuntimeError("Load models before calling get_expert().")
        return self.model_dict[diameter]

    def load_models(self, path: str) -> None:
        """Load all diameter-specific experts from a checkpoint file.

        Args:
            path: Path to a checkpoint mapping each diameter to a tuple of
                model weights, input scaler state, and output scaler state.
        """
        multi_expert_operator_state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        for diameter in self.diameters:
            state_dict, scaler_input_state_dict, scaler_output_state_dict = (
                multi_expert_operator_state_dict[diameter]
            )
            shape = state_dict["net.Layer_0.weight"].shape
            hidden_size = shape[0]
            n_frequencies = shape[1] // 2
            # Reconstruct the saved architecture from its first linear layer.
            layers = [16, hidden_size, 2]
            model = SaintVenantNeuralOperator(
                layers,
                num_frequencies=n_frequencies,
                scale=1e10,
                activation_function=nn.ReLU,
                isLayerNormalizationActivated=True,
            )
            model.load_state_dict(state_dict)
            model.scaler_input = self._load_scaler(scaler_input_state_dict)
            model.scaler_output = self._load_scaler(scaler_output_state_dict)
            self.model_dict[diameter] = model
        self.is_model_loaded = True

    @staticmethod
    def _load_scaler(scaler: ScalerStandard | dict[str, Tensor]) -> ScalerStandard:
        """Normalize legacy dict scaler states into ``ScalerStandard`` objects."""
        if isinstance(scaler, dict):
            return ScalerStandard.load(scaler)
        return scaler

    def to_device(self, device: torch.device = torch.device("cuda")) -> None:
        """Move all loaded models and their scalers to ``device``."""
        for model in self.model_dict.values():
            model.to(device)
            model.scaler_input.to(device)
            model.scaler_output.to(device)

    def predict_one(
        self,
        diameter: int,
        length: float,
        slope: float,
        roughness: float,
        left_boundary_conditions: NDArray | Tensor,
        right_boundary_conditions: NDArray | Tensor,
        x: float,
    ) -> Tensor: 
        """Prediction one data point. This is much slower than `predict_batch` but has more validation checks.

        Args:
            diameter (int): _description_
            length (float): _description_
            slope (float): _description_
            roughness (float): _description_
            left_boundary_conditions (NDArray | Tensor): _description_
            right_boundary_conditions (NDArray | Tensor): _description_
            x (float): _description_

        Returns:
            Tensor: _description_
        """
        assert self.is_model_loaded, "Load models before calling predict()."
        assert left_boundary_conditions.shape[0] == 6, "Left boundary conditions should have length 6."
        assert right_boundary_conditions.shape[0] == 6, "Right boundary conditions should have length 6."
        self.check_nominal_diameter(diameter)
        model = self.get_expert(diameter) # select an expert based on the diameter
        features = torch.tensor(
            [
                length,
                slope,
                roughness,
                *left_boundary_conditions.tolist(),
                *right_boundary_conditions.tolist(),
                x,
            ],
            dtype=torch.float32,
        )
        return model.predict(features)

    def predict_batch(self, diameter: int, features: Tensor) -> Tensor:
        self.check_nominal_diameter(diameter)
        model = self.get_expert(diameter) # select an expert based on the diameter
        return model.predict(features)

    def check_nominal_diameter(self, diameter: int) -> None:
        """Check if the given diameter is supported by the loaded models."""
        if diameter not in self.diameters:
            raise ValueError(f"Diameter {diameter} mm is not a nominal diameter.")