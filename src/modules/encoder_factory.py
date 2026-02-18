"""
Encoder factory for creating different encoder types.

Supported encoder types:
- "transformer_v1": Standard nn.TransformerEncoder (default, matches original implementation)
- "conformer": torchaudio.models.Conformer with adapter for interface compatibility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.conformer import Conformer


def _create_transformer_v1_encoder(
    d_model: int,
    params: dict,
    device: str | torch.device | None = None
) -> nn.TransformerEncoder:
    """
    Creates a standard Transformer encoder.

    Arguments:
    ----------
    d_model: int
        Model dimension
    params: dict
        Dictionary with encoder parameters:
        - num_layers: Number of encoder layers (default: 4)
        - num_heads: Number of attention heads (default: 4)
        - dim_feedforward: Feed-forward network dimension (default: 128)
        - dropout: Dropout probability (default: 0.1)
    device: str | torch.device | None
        Device to create the encoder on

    Returns:
    --------
    encoder: nn.TransformerEncoder
        Standard Transformer encoder instance
    """
    DEFAULT_NUM_LAYERS = 4
    DEFAULT_NUM_HEADS = 4
    DEFAULT_DIM_FEEDFORWARD = 128
    DEFAULT_DROPOUT = 0.1

    num_encoder_layers = params.get("num_layers", DEFAULT_NUM_LAYERS)
    num_heads_encoder = params.get("num_heads", DEFAULT_NUM_HEADS)
    dim_feedforward = params.get("dim_feedforward", DEFAULT_DIM_FEEDFORWARD)
    dropout = params.get("dropout", DEFAULT_DROPOUT)
    activation = F.relu

    encoder_norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads_encoder,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )

    encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_encoder_layers,
        norm=encoder_norm,
    )

    if device is not None:
        encoder = encoder.to(device)

    return encoder


def _create_conformer_encoder(
    d_model: int,
    params: dict,
    device: str | torch.device | None = None
) -> nn.Module:
    """
    Creates a Conformer encoder

    Arguments:
    ----------
    d_model: int
        Model dimension (input_dim for Conformer)
    params: dict
        Dictionary with encoder parameters:
        - num_heads: Number of attention heads (default: 4)
        - ffn_dim: Feed-forward network dimension (default: 128)
        - num_layers: Number of Conformer layers (default: 4)
        - depthwise_conv_kernel_size: Kernel size for depthwise convolution (default: 31)
        - dropout: Dropout probability (default: 0.0)
        - use_group_norm: Use GroupNorm instead of BatchNorm (default: False)
        - convolution_first: Apply convolution before attention (default: False)
    device: str | torch.device | None
        Device to create the encoder on

    Returns:
    --------
    encoder: nn.Module
        Conformer
    """
    DEFAULT_NUM_HEADS = 4
    DEFAULT_FFN_DIM = 128
    DEFAULT_NUM_LAYERS = 4
    DEFAULT_DEPTHWISE_CONV_KERNEL_SIZE = 31
    DEFAULT_DROPOUT = 0.0
    DEFAULT_USE_GROUP_NORM = False
    DEFAULT_CONVOLUTION_FIRST = False

    num_heads = params.get("num_heads", DEFAULT_NUM_HEADS)
    ffn_dim = params.get("ffn_dim", DEFAULT_FFN_DIM)
    num_layers = params.get("num_layers", DEFAULT_NUM_LAYERS)
    depthwise_conv_kernel_size = params.get("depthwise_conv_kernel_size", DEFAULT_DEPTHWISE_CONV_KERNEL_SIZE)
    dropout = params.get("dropout", DEFAULT_DROPOUT)
    use_group_norm = params.get("use_group_norm", DEFAULT_USE_GROUP_NORM)
    convolution_first = params.get("convolution_first", DEFAULT_CONVOLUTION_FIRST)

    conformer = Conformer(
        input_dim=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        dropout=dropout,
        use_group_norm=use_group_norm,
        convolution_first=convolution_first,
    )

    if device is not None:
        conformer = conformer.to(device)

    return conformer


def encoder_factory(
    config: dict,
    d_model: int,
    device: str | torch.device | None = None
) -> nn.Module:
    """
    Creates an encoder from configuration.

    Config structure:
    {
        "type": "transformer_v1" | "conformer",
        "params": {
            // type-specific parameters (optional, defaults will be used)
        }
    }

    Arguments:
    ----------
    config: dict
        Encoder configuration dictionary with keys:
        - type: str, encoder type ("transformer_v1", "conformer", etc.)
        - params: dict, type-specific parameters (optional)
    d_model: int
        Model dimension
    device: str | torch.device | None
        Device to create the encoder on

    Returns:
    --------
    encoder: nn.Module
        Encoder module compatible with our interface:
        encoder(x, src_key_padding_mask) -> encoded

    Raises:
    -------
    ValueError
        If encoder type is unknown
    TypeError
        If config is not a dictionary
    """
    if not isinstance(config, dict):
        raise TypeError(f"encoder config must be a dict, got {type(config)}")

    if "type" not in config:
        raise ValueError("encoder config must have 'type' key")

    encoder_type = config["type"]
    params = config.get("params", {})

    if encoder_type == "transformer_v1":
        return _create_transformer_v1_encoder(d_model, params, device)
    elif encoder_type == "conformer":
        return _create_conformer_encoder(d_model, params, device)
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Supported types: 'transformer_v1', 'conformer'"
        )
