
import torch
import torch.nn as nn

from modules.positional_encodings import SinusoidalPositionalEncoding
from modules.swipe_point_embedder_factory import swipe_point_embedder_factory
from modules.encoder_factory import encoder_factory
from modules.decoder_factory import decoder_factory



def _get_mask(max_seq_len: int):
    """
    Returns a mask for the decoder transformer.
    """
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask



# encode() and decode() methods are extremely useful in decoding algorithms
# like beamsearch where we do encoding once and decoding multimple times.
# This reduces computations up to two times.
class EncoderDecoderTransformerLike(nn.Module):
    def _get_mask(self, max_seq_len: int):
        """
        Returns a mask for the decoder transformer.
        """
        return _get_mask(max_seq_len)

    def __init__(self,
                 enc_in_emb_model: nn.Module,
                 dec_in_emb_model: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 out: nn.Module,
                 device: str | None = None):
        super().__init__()
        self.enc_in_emb_model = enc_in_emb_model
        self.dec_in_emb_model = dec_in_emb_model
        self.encoder = encoder
        self.decoder = decoder
        self.out = out  # linear
        self.device = torch.device(
            device or 'cuda' if torch.cuda.is_available() else 'cpu')

    # x can be a tuple (ex. traj_feats, kb_tokens) or a single tensor
    # (ex. just kb_tokens).
    def encode(self, tupled_x, x_pad_mask):
        x = self.enc_in_emb_model(*tupled_x)
        return self.encoder(x, src_key_padding_mask = x_pad_mask)

    def decode(self, y, x_encoded, memory_key_padding_mask, tgt_key_padding_mask):
        y = self.dec_in_emb_model(y)
        tgt_mask = self._get_mask(len(y)).to(device=self.device)
        dec_out = self.decoder(y, x_encoded, tgt_mask=tgt_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out(dec_out)

    def forward(self, tupled_x, y, x_pad_mask, y_pad_mask):
        x_encoded = self.encode(tupled_x, x_pad_mask)
        return self.decode(y, x_encoded, x_pad_mask, y_pad_mask)



################################################################################


def get_word_char_embedder__vn1(d_model: int,
                                n_word_chars: int,
                                max_out_seq_len: int=35,
                                dropout: float=0.1,
                                device=None) -> nn.Module:
    """
    Creates a word character embedding model with positional encoding.

    Arguments:
    ----------
    d_model: int
        Model dimension
    n_word_chars: int
        Number of word characters (vocabulary size)
    max_out_seq_len: int
        Maximum output sequence length (default: 35)
    dropout: float
        Dropout probability (default: 0.1)
    device: str | torch.device | None
        Device to create the model on

    Returns:
    --------
    word_char_embedding_model: nn.Module
        Sequential model with embedding, dropout, and positional encoding
    """
    word_char_embedding = nn.Embedding(n_word_chars, d_model)
    word_char_emb_dropout = nn.Dropout(dropout)
    word_char_pos_encoder = SinusoidalPositionalEncoding(d_model, max_out_seq_len, device=device)

    word_char_embedding_model = nn.Sequential(
        word_char_embedding,
        word_char_emb_dropout,
        word_char_pos_encoder
    )

    return word_char_embedding_model


def _get_device(device: torch.device | str | None = None) -> torch.device:
    """
    Returns the input if not None, otherwise returns the default device.
    Default device is 'cuda' if available, otherwise 'cpu'.

    Arguments:
    ----------
    device: torch.device | str | None
        Device to use

    Returns:
    --------
    device: torch.device
        Torch device object
    """
    return torch.device(
        device
        or 'cuda' if torch.cuda.is_available() else 'cpu'
    )


def _set_state(model: nn.Module,
               weights_path: str,
               device: torch.device | str | None = None
               ) -> nn.Module:
    """
    Sets the state of the model from the weights_path.
    If weights_path is None, the model is returned without loading any state.

    Arguments:
    ----------
    model: nn.Module
        Model to load weights into
    weights_path: str
        Path to weights file (optional)
    device: torch.device | str | None
        Device to load the model onto

    Returns:
    --------
    model: nn.Module
        Model with loaded weights (if provided) and on the correct device
    """
    if weights_path:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model = model.eval()
    return model


def get_model_from_configs(
    input_embedding_config: dict,
    encoder_config: dict,
    decoder_config: dict,
    n_classes: int,
    n_word_tokens: int,
    max_out_seq_len: int,
    d_model: int,
    device: torch.device | str | None = None,
    weights_path: str | None = None
) -> EncoderDecoderTransformerLike:
    """
    Meta-factory that assembles a complete model from component configs.

    This is the primary entry point for creating models. It calls all the
    individual factory functions and returns a fully assembled model.

    Arguments:
    ----------
    input_embedding_config: dict
        Configuration for the input embedding (e.g., swipe point embedder)
        Format: {"type": "...", "params": {...}}
    encoder_config: dict
        Configuration for the encoder
        Format: {"type": "...", "params": {...}}
    decoder_config: dict
        Configuration for the decoder
        Format: {"type": "...", "params": {...}}
    n_classes: int
        Number of output classes
    n_word_tokens: int
        Number of word tokens (vocabulary size for decoder input)
    max_out_seq_len: int
        Maximum output sequence length
    d_model: int
        Model dimension (must match the output dimension of the swipe point embedder)
    device: torch.device | str | None
        Device to create the model on (default: cuda if available, else cpu)
    weights_path: str | None
        Path to pretrained weights (optional)

    Returns:
    --------
    model: EncoderDecoderTransformerLike
        Fully assembled model ready for training or inference
    """
    device = _get_device(device)

    # Create all components via factories
    input_embedding = swipe_point_embedder_factory(input_embedding_config, device)
    encoder = encoder_factory(encoder_config, d_model, device)
    decoder = decoder_factory(decoder_config, d_model, device)
    word_char_embedding_model = get_word_char_embedder__vn1(
        d_model, n_word_tokens, max_out_seq_len=max_out_seq_len,
        dropout=0.1, device=device)
    output_proj = nn.Linear(d_model, n_classes, device=device)

    # Assemble model
    model = EncoderDecoderTransformerLike(
        input_embedding,
        word_char_embedding_model,
        encoder,
        decoder,
        output_proj,
        device=device
    )

    # Load weights if provided
    model = _set_state(model, weights_path, device)

    return model
