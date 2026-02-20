"""
Export model via executorch for inference on Android.
"""
import sys; from pathlib import Path; sys.path.insert(1, str(Path.cwd() / "src"))

import argparse
import json
from pathlib import Path

import torch
from torch.export import export, ExportedProgram, Dim
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

from .model import get_model_from_configs, EncoderDecoderTransformerLike
from .ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizer
from .feature_extraction.swipe_feature_extractor_factory import swipe_feature_extractor_factory
from .feature_extraction.swipe_feature_extractors import MultiFeatureExtractor
from .train import validate_d_model


def read_json(path: str) -> dict:
    """Read JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export model to executorch format")
    parser.add_argument(
        '--checkpoint_path', type=str, required=True,
        help='Path to the model checkpoint (.ckpt or .pt file). If not provided, uses model_weights_path from prediction_config'
    )
    parser.add_argument(
        '--train_config', type=str, required=True,
        help='Path to the training config JSON file (saved in lightning_logs/version_x/config.json)'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save exported .pte file (default: ./results/executorch_models/{checkpoint_stem}__{backend}.pte)'
    )
    parser.add_argument(
        '--backend', type=str, default='xnnpack', choices=['xnnpack', 'raw'],
        help='Backend to use for export (default: xnnpack)'
    )

    args = parser.parse_args()

    # Set default output_path if not provided
    if args.output_path is None:
        checkpoint_stem = Path(args.checkpoint_path).stem
        args.output_path = f'./results/executorch_models/{checkpoint_stem}__{args.backend}.pte'

    return args


class Encode(torch.nn.Module):
    """Encoder wrapper for executorch export."""

    def __init__(self, model: EncoderDecoderTransformerLike) -> None:
        super().__init__()
        self.enc_in_emb_model = model.enc_in_emb_model
        self.encoder = model.encoder

    def forward(self, encoder_in):
        x = self.enc_in_emb_model(*encoder_in)
        result = self.encoder(x, src_key_padding_mask=None)
        return result


class Decode(torch.nn.Module):
    """Decoder wrapper for executorch export."""

    def __init__(self, model: EncoderDecoderTransformerLike) -> None:
        super().__init__()
        self.dec_in_emb_model = model.dec_in_emb_model
        self.decoder = model.decoder
        self._get_mask = model._get_mask
        self.out = model.out

    def forward(self, decoder_in, x_encoded):
        y = self.dec_in_emb_model(decoder_in)
        tgt_mask = self._get_mask(y.size(0))
        dec_out = self.decoder(
            y, x_encoded, tgt_mask=tgt_mask,
            memory_key_padding_mask=None,
            tgt_key_padding_mask=None,
            tgt_is_causal=True)
        return self.out(dec_out)


def create_swipe_feature_extractor_from_config(config: dict) -> MultiFeatureExtractor:
    grids = read_json(config['grids_path'])
    grid = grids[config['grid_name']]
    trajectory_stats = read_json(config['trajectory_features_statistics_path'])
    bounding_boxes = read_json(config['bounding_boxes_path'])
    feature_extractor = swipe_feature_extractor_factory(
        grid=grid,
        keyboard_tokenizer=KeyboardTokenizer(config['keyboard_tokenizer_path']),
        trajectory_features_statistics=trajectory_stats,
        bounding_boxes=bounding_boxes,
        grid_name=config['grid_name'],
        component_configs=config['feature_extractor']
    )
    return feature_extractor



def create_model_from_config(config: dict,
                             d_model: int,
                             device: str,
                             word_tokenizer: CharLevelTokenizerv2,
                             checkpoint_path: str) -> EncoderDecoderTransformerLike:

    model = get_model_from_configs(
        input_embedding_config=config['swipe_point_embedder'],
        encoder_config=config['encoder'],
        decoder_config=config['decoder'],
        n_classes=config['num_classes'],
        n_word_tokens=len(word_tokenizer.char_to_idx),
        max_out_seq_len=config['max_out_seq_len'],
        d_model=d_model,
        device=device,
        weights_path=checkpoint_path
    )
    return model


def create_sample_inputs(
    feature_extractor: MultiFeatureExtractor,
    word_tokenizer: CharLevelTokenizerv2,
    swipe_length: int = 13,
    sample_word: str = "test"
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """
    Returns:
    --------
    encoder_in: tuple of Tensors
        The encoder input (swipe_features from feature extractor)
    decoder_in: Tensor
        The decoder input (tokenized sample word)
    """
    # Generate sample raw swipe data (x, y, t coordinates)
    # Simulating a diagonal swipe from top-left to bottom-right
    x = torch.tensor([500 + i * 20 for i in range(swipe_length)], dtype=torch.float32)
    y = torch.tensor([100 + i * 15 for i in range(swipe_length)], dtype=torch.float32)
    t = torch.tensor([i * 40 for i in range(swipe_length)], dtype=torch.float32)

    # Extract features using the feature extractor
    swipe_features = feature_extractor(x, y, t)

    # Add batch dimension to encoder inputs (batch_first=False, so unsqueeze at dim 1)
    encoder_in = tuple(feat.unsqueeze(1) for feat in swipe_features)

    # Create decoder input from sample word
    tgt_token_seq = word_tokenizer.encode(sample_word)
    tgt_token_seq = torch.tensor(tgt_token_seq, dtype=torch.int64)
    decoder_in = tgt_token_seq[:-1]  # Remove last token for decoder input
    decoder_in = decoder_in.unsqueeze(1)  # Add batch dimension

    return tuple(encoder_in), decoder_in


def verify_exported_model(
    model: EncoderDecoderTransformerLike,
    encoder_in: tuple[torch.Tensor, ...],
    decoder_in: torch.Tensor,
    pte_path: str
) -> None:
    """
    Verify that the exported ExecuTorch model produces similar results to the PyTorch model.

    Arguments:
    ----------
    model: EncoderDecoderTransformerLike
        The original PyTorch model
    encoder_in: tuple of Tensors
        Sample encoder inputs
    decoder_in: Tensor
        Sample decoder input
    pte_path: str
        Path to the exported .pte file
    """
    print("\nVerifying exported model...")

    with torch.no_grad():
        pytorch_encoded = model.encode(encoder_in, None)
        pytorch_decoded = model.decode(decoder_in, pytorch_encoded, None, None)

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)

    encode_method = program.load_method("encode")
    executorch_encoded_list = encode_method.execute(list(encoder_in))
    executorch_encoded = executorch_encoded_list[0]

    decode_method = program.load_method("decode")
    executorch_decoded_list = decode_method.execute([decoder_in, executorch_encoded])
    executorch_decoded = executorch_decoded_list[0]

    encoder_match = torch.allclose(executorch_encoded, pytorch_encoded, rtol=1e-3, atol=1e-5)
    print(f"Encoder outputs match: {encoder_match}")
    if not encoder_match:
        max_diff = (executorch_encoded - pytorch_encoded).abs().max().item()
        print(f"  Max encoder difference: {max_diff}")

    decoder_match = torch.allclose(executorch_decoded, pytorch_decoded, rtol=1e-3, atol=1e-5)
    print(f"Decoder outputs match: {decoder_match}")
    if not decoder_match:
        max_diff = (executorch_decoded - pytorch_decoded).abs().max().item()
        print(f"  Max decoder difference: {max_diff}")

    if encoder_match and decoder_match:
        print("Verification PASSED: Exported model produces similar results to PyTorch model")
    else:
        print("Verification WARNING: Outputs differ beyond tolerance")


def main():
    args = parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    train_config = read_json(args.train_config)

    feature_extractor = create_swipe_feature_extractor_from_config(train_config)
    word_tokenizer = CharLevelTokenizerv2(train_config['vocab_path'])

    d_model = train_config.get('d_model')
    if d_model is None:
        raise ValueError(
            "d_model must be specified in config. "
            "It should match the output dimension of the swipe point embedder."
        )
    validate_d_model(d_model, feature_extractor, train_config['swipe_point_embedder'])    

    model = create_model_from_config(train_config, 
                                     d_model=d_model, 
                                     device='cpu', 
                                     word_tokenizer=word_tokenizer,
                                     checkpoint_path=args.checkpoint_path)
    model.eval()

    print(f"Model loaded successfully")

    encoder_in, decoder_in = create_sample_inputs(feature_extractor, word_tokenizer)

    encoded = model.encode(encoder_in, None)

    print(f"Sample inputs created")
    print(f"  encoder_in shapes: traj_feats={encoder_in[0].shape}, kb_ids={encoder_in[1].shape}")
    print(f"  decoder_in shape: {decoder_in.shape}")
    print(f"  encoded shape: {encoded.shape}")


    # Define dynamic shapes for export
    MAX_SWIPE_LEN = 299
    MAX_WORD_LEN = 35
    dim_swipe_seq = Dim("dim_swipe_seq", min=1, max=MAX_SWIPE_LEN)
    dim_char_seq = Dim("dim_char_seq", min=1, max=MAX_WORD_LEN)

    # For encoder: encoder_in is a single arg that is a tuple of N tensors
    # `dynamic_shapes` needs to be a 1-element tuple containing the N-element tuple of shapes
    encoder_dynamic_shapes = (tuple({0: dim_swipe_seq} for _ in encoder_in),)

    decoder_dynamic_shapes = (
        {0: dim_char_seq},  # decoder_in
        {0: dim_swipe_seq}   # x_encoded
    )

    print("Exporting encoder...")
    aten_encode: ExportedProgram = export(
        Encode(model).eval(),
        (encoder_in,),
        dynamic_shapes=encoder_dynamic_shapes
    )

    print("Exporting decoder...")
    aten_decode: ExportedProgram = export(
        Decode(model).eval(),
        (decoder_in, encoded),
        dynamic_shapes=decoder_dynamic_shapes
    )

    # Export to executorch
    if args.backend == "xnnpack":
        print("Exporting with XNNPACK backend...")
        edge_xnnpack = to_edge_transform_and_lower(
            {"encode": aten_encode, "decode": aten_decode},
            partitioner=[XnnpackPartitioner()],
        )
        exec_prog = edge_xnnpack.to_executorch()

        with open(args.output_path, "wb") as file:
            exec_prog.write_to_file(file)
        print(f"XNNPACK model saved to {args.output_path}")
    else:  # raw
        print("Exporting raw executorch model...")
        edge_program = to_edge({"encode": aten_encode, "decode": aten_decode})
        exec_prog = edge_program.to_executorch()

        with open(args.output_path, "wb") as file:
            file.write(exec_prog.buffer)
        print(f"Raw model saved to {args.output_path}")

    print("Export complete!")

    verify_exported_model(model, encoder_in, decoder_in, args.output_path)


if __name__ == "__main__":
    main()
