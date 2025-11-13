import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import os
import json
import argparse

import logging

from predict import predict


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for all model checkpoints in the config's directory.")
    parser.add_argument('--config', type=str, required=True, help='Path to the base prediction config file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for prediction (e.g., cuda, cpu).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of worker processes for prediction.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_config = json.load(f)

    # Assume checkpoints are in the same directory as the config file
    ckpt_dir = os.path.dirname(os.path.abspath(base_config['model_weights_path']))
    
    checkpoint_files = [os.path.join(ckpt_dir, f)
                        for f in os.listdir(ckpt_dir)
                        if f.endswith(('.pt'))]

    if not checkpoint_files:
        raise FileNotFoundError(f"No .pt files found in the checkpoint directory: {ckpt_dir}")

    output_dir = os.path.dirname(base_config['output_path'])
    os.makedirs(output_dir, exist_ok=True)

    for ckpt_path in checkpoint_files:
        logging.info(f"Processing checkpoint: {ckpt_path}")

        # Generate a clean output filename from the checkpoint name
        ckpt_filename_no_ext = os.path.basename(ckpt_path).rsplit('.', 1)[0]
        output_filename = f"{ckpt_filename_no_ext}.pkl"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            logging.info(f"Output file already exists, skipping: {output_path}")
            continue

        # Create a config for the current checkpoint in memory
        temp_config = base_config.copy()
        temp_config['model_weights_path'] = ckpt_path
        temp_config['output_path'] = output_path
        temp_config['device'] = args.device

        logging.info(f"Temporary config for {ckpt_path}: {temp_config}")
        
        # Run the prediction function directly
        try:
            logging.info(f"Running predictions for {ckpt_path}...")
            predict(temp_config, args.num_workers)
            logging.info(f"Finished predictions. Output saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to process checkpoint {ckpt_path}: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
