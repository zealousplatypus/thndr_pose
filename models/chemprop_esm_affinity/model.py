"""Chemprop model construction for the ESM-conditioned affinity baseline."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from .config import ExperimentConfig


def _chemprop_model_modules() -> tuple[Any, Any]:
    """Import Chemprop model modules lazily."""
    try:
        chemprop_nn = importlib.import_module("chemprop.nn")
        chemprop_models = importlib.import_module("chemprop.models")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chemprop is required for model construction. Install Chemprop v2 "
            "in this environment before running training."
        ) from exc
    return chemprop_nn, chemprop_models


def _call_with_supported_kwargs(callable_obj: Any, **kwargs: Any) -> Any:
    """Call a Chemprop constructor with only supported keyword arguments."""
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable_obj(**kwargs)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return callable_obj(**supported)


def _transform_from_scaler(transform_cls: Any, scaler: Any | None) -> Any | None:
    """Build a Chemprop transform from a fitted scaler when supported."""
    if scaler is None or transform_cls is None:
        return None
    if hasattr(transform_cls, "from_standard_scaler"):
        return transform_cls.from_standard_scaler(scaler)
    if hasattr(transform_cls, "from_scaler"):
        return transform_cls.from_scaler(scaler)
    return _call_with_supported_kwargs(transform_cls, scaler=scaler)


def _build_message_passing(config: ExperimentConfig, chemprop_nn: Any) -> Any:
    """Build the Chemprop bond message-passing module."""
    mp_config = config.model.message_passing
    return _call_with_supported_kwargs(
        chemprop_nn.BondMessagePassing,
        d_h=mp_config.hidden_dim,
        hidden_dim=mp_config.hidden_dim,
        depth=mp_config.depth,
        dropout=mp_config.dropout,
        undirected=mp_config.undirected,
        activation=mp_config.activation,
    )


def _build_aggregation(config: ExperimentConfig, chemprop_nn: Any) -> Any:
    """Build the Chemprop aggregation module."""
    if config.model.aggregation != "mean":
        raise ValueError("Only mean aggregation is supported for this baseline")
    return chemprop_nn.MeanAggregation()


def _build_regression_ffn(
    config: ExperimentConfig,
    chemprop_nn: Any,
    input_dim: int,
    target_scaler: Any | None,
) -> Any:
    """Build a regression FFN with optional target unscale transform."""
    ffn_config = config.model.ffn
    output_transform = None
    if target_scaler is not None and hasattr(chemprop_nn, "UnscaleTransform"):
        output_transform = _transform_from_scaler(chemprop_nn.UnscaleTransform, target_scaler)

    return _call_with_supported_kwargs(
        chemprop_nn.RegressionFFN,
        input_dim=input_dim,
        d_v=input_dim,
        hidden_dim=ffn_config.hidden_dim,
        d_h=ffn_config.hidden_dim,
        n_layers=ffn_config.num_layers,
        num_layers=ffn_config.num_layers,
        dropout=ffn_config.dropout,
        output_transform=output_transform,
    )


def build_chemprop_model(
    config: ExperimentConfig,
    esm_dim: int,
    target_scaler: Any | None = None,
    descriptor_scaler: Any | None = None,
) -> Any:
    """Build the Chemprop MPNN that consumes ESM descriptors as `X_d`."""
    chemprop_nn, chemprop_models = _chemprop_model_modules()
    message_passing = _build_message_passing(config, chemprop_nn)
    aggregation = _build_aggregation(config, chemprop_nn)
    mp_output_dim = int(getattr(message_passing, "output_dim", config.model.message_passing.hidden_dim))
    ffn_input_dim = mp_output_dim + int(esm_dim)
    ffn = _build_regression_ffn(config, chemprop_nn, ffn_input_dim, target_scaler)

    x_d_transform = None
    if descriptor_scaler is not None and hasattr(chemprop_nn, "ScaleTransform"):
        x_d_transform = _transform_from_scaler(chemprop_nn.ScaleTransform, descriptor_scaler)

    return _call_with_supported_kwargs(
        chemprop_models.MPNN,
        message_passing=message_passing,
        mp=message_passing,
        agg=aggregation,
        aggregation=aggregation,
        predictor=ffn,
        ffn=ffn,
        batch_norm=config.model.batch_norm,
        X_d_transform=x_d_transform,
        x_d_transform=x_d_transform,
    )

