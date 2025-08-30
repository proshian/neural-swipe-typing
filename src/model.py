from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.positional_encodings import SinusoidalPositionalEncoding
from modules.swipe_point_embedder_factory import swipe_point_embedder_factory


D_MODEL_V1 = 128



def _get_mask(max_seq_len: int):
    """
    Returns a mask for the decoder transformer.
    """
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask



# encode() and decode() methods are extremely useful in decoding algorithms
# like beamsearch where we do encoding once and decoding multimple times.
# Rhis reduces computations up to two times.
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
                 device: Optional[str] = None):
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



def get_transformer_encoder_backbone__vn1() -> nn.TransformerEncoder:
    
    num_encoder_layers = 4
    num_heads_encoder = 4
    dim_feedforward = 128
    dropout = 0.1
    activation = F.relu

    encoder_norm = nn.LayerNorm(D_MODEL_V1, eps=1e-5, bias=True)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL_V1,
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

    return encoder


def get_transformer_decoder_backbone__vn1() -> nn.TransformerDecoder:
    
    num_decoder_layers = 4
    num_heads_decoder = 4
    dim_feedforward = 128
    dropout = 0.1
    activation = F.relu

    decoder_norm = nn.LayerNorm(D_MODEL_V1, eps=1e-5, bias=True)

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=D_MODEL_V1,
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

    return decoder
                                                


def get_word_char_embedder__vn1(d_model: int, 
                                n_word_chars: int, 
                                max_out_seq_len: int=35, 
                                dropout: float=0.1,
                                device=None) -> nn.Module:
    word_char_embedding = nn.Embedding(n_word_chars, d_model)
    dropout = 0.1
    word_char_emb_dropout = nn.Dropout(dropout)
    word_char_pos_encoder = SinusoidalPositionalEncoding(d_model, max_out_seq_len, device=device)

    word_char_embedding_model = nn.Sequential(
        word_char_embedding,
        word_char_emb_dropout,
        word_char_pos_encoder
    )

    return word_char_embedding_model


def _get_device(device: Optional[Union[torch.device, str]] = None) -> torch.device:
    """
    Returns the input if not None, otherwise returns the default device.
    Default device is 'cuda' if available, otherwise 'cpu'.
    """
    return torch.device(
        device 
        or 'cuda' if torch.cuda.is_available() else 'cpu'
    )


def _get_transformer__vn1(input_embedding: nn.Module,
                          n_classes: int,
                          n_word_tokens: int,
                          max_out_seq_len: int,
                          device = None,):
    device = _get_device(device)

    word_char_embedding_model = get_word_char_embedder__vn1(
        D_MODEL_V1, n_word_tokens, max_out_seq_len=max_out_seq_len,
        dropout=0.1, device=device)

    out = nn.Linear(D_MODEL_V1, n_classes, device = device)

    encoder = get_transformer_encoder_backbone__vn1()
    decoder = get_transformer_decoder_backbone__vn1()

    return EncoderDecoderTransformerLike(
        input_embedding, word_char_embedding_model, encoder, decoder, out
    )



def _set_state(model: nn.Module, 
               weights_path: str, 
               device: Optional[Union[torch.device, str]] = None
               ) -> nn.Module:
    """
    Sets the state of the model from the weights_path.
    If weights_path is None, the model is returned without loading any state.
    """
    if weights_path:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model = model.eval()
    return model



def get_transformer__from_spe_config__vn1(spe_config: dict,
                                          n_classes: int,
                                          n_word_tokens: int,
                                          max_out_seq_len: int,
                                          device: Optional[Union[torch.device, str]] = None,
                                          weights_path: str = None
                                          ) -> EncoderDecoderTransformerLike:
    input_embedding = swipe_point_embedder_factory(spe_config)
    model = _get_transformer__vn1(
        input_embedding, n_classes, n_word_tokens, max_out_seq_len, device)
    model = _set_state(model, weights_path, device)
    return model
