"""
Decoder factory for creating different decoder types.

Supported decoder types:
- "transformer_v1": Standard nn.TransformerDecoder (matches original implementation)
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_transformer_v1_decoder(
    d_model: int,
    params: dict,
    device: Optional[Union[str, torch.device]] = None
) -> nn.TransformerDecoder:
    """
    Creates a standard Transformer decoder matching the original v1 implementation.

    Arguments:
    ----------
    d_model: int
        Model dimension
    params: dict
        Dictionary with decoder parameters:
        - num_layers: Number of decoder layers (default: 4)
        - num_heads: Number of attention heads (default: 4)
        - dim_feedforward: Feed-forward network dimension (default: 128)
        - dropout: Dropout probability (default: 0.1)
    device: Optional[Union[str, torch.device]]
        Device to create the decoder on

    Returns:
    --------
    decoder: nn.TransformerDecoder
        Standard Transformer decoder instance
    """
    DEFAULT_NUM_LAYERS = 4
    DEFAULT_NUM_HEADS = 4
    DEFAULT_DIM_FEEDFORWARD = 128
    DEFAULT_DROPOUT = 0.1

    num_decoder_layers = params.get("num_layers", DEFAULT_NUM_LAYERS)
    num_heads_decoder = params.get("num_heads", DEFAULT_NUM_HEADS)
    dim_feedforward = params.get("dim_feedforward", DEFAULT_DIM_FEEDFORWARD)
    dropout = params.get("dropout", DEFAULT_DROPOUT)
    activation = F.relu

    decoder_norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads_decoder,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )

    decoder = nn.TransformerDecoder(
        decoder_layer,
        num_layers=num_decoder_layers,
        norm=decoder_norm,
    )

    if device is not None:
        decoder = decoder.to(device)

    return decoder


def decoder_factory(
    config: dict,
    d_model: int,
    device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Creates a decoder from configuration.

    Config structure:
    {
        "type": "transformer_v1",
        "params": {
            // type-specific parameters (optional, defaults will be used)
        }
    }

    Arguments:
    ----------
    config: dict
        Decoder configuration dictionary with keys:
        - type: str, decoder type ("transformer_v1")
        - params: dict, type-specific parameters (optional)
    d_model: int
        Model dimension
    device: Optional[Union[str, torch.device]]
        Device to create the decoder on

    Returns:
    --------
    decoder: nn.Module
        Decoder module compatible with our interface

    Raises:
    ------
    ValueError
        If decoder type is unknown
    TypeError
        If config is not a dictionary
    """
    if not isinstance(config, dict):
        raise TypeError(f"decoder config must be a dict, got {type(config)}")

    if "type" not in config:
        raise ValueError("decoder config must have 'type' key")

    decoder_type = config["type"]
    params = config.get("params", {})

    if decoder_type == "transformer_v1":
        return _create_transformer_v1_decoder(d_model, params, device)
    else:
        raise ValueError(
            f"Unknown decoder type: {decoder_type}. "
            f"Supported types: 'transformer_v1'"
        )
