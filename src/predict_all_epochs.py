"""Run predict.py for all checkpoints in the directory containing model_weights_path.

Usage: python src/predict_all_epochs.py model_weights_path=<path> [other predict.py args...]
"""
import logging
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def find_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Find all .pt and .ckpt files in the directory, sorted by modification time."""
    checkpoints = list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.ckpt"))
    checkpoints = sorted(checkpoints, key=lambda p: p.stat().st_mtime)
    return checkpoints


def get_output_dir(train_cfg: dict, cfg: DictConfig) -> str:
    """Get output directory: results/predictions/{exp_name}/{data_name}/{grid_name}/"""
    exp_name = train_cfg.get("experiment_name", "unknown")
    data_name = Path(cfg.data_path).stem
    grid_name = train_cfg.get("grid_name", "default")
    return f"results/predictions/{exp_name}/{data_name}/{grid_name}"


@hydra.main(version_base=None, config_path="../configs", config_name="predict_from_train")
def main(cfg: DictConfig) -> None:
    original_cwd = get_original_cwd()

    if not cfg.model_weights_path:
        print("Error: model_weights_path is required")
        sys.exit(1)
    if not cfg.train_config_path:
        print("Error: train_config_path is required")
        sys.exit(1)
    if not cfg.data_path:
        print("Error: data_path is required")
        sys.exit(1)

    # Resolve paths for filesystem operations
    model_weights_path = cfg.model_weights_path
    train_config_path = cfg.train_config_path

    if not Path(model_weights_path).is_absolute():
        model_weights_path = str(Path(original_cwd) / model_weights_path)
    if not Path(train_config_path).is_absolute():
        train_config_path = str(Path(original_cwd) / train_config_path)

    # Load train config to get experiment_name and grid_name
    train_cfg = OmegaConf.to_container(OmegaConf.load(train_config_path), resolve=True)
    output_dir = get_output_dir(train_cfg, cfg)

    ckpt_dir = Path(model_weights_path).parent
    checkpoints = find_checkpoints(ckpt_dir)

    if not checkpoints:
        log.error(f"No .pt or .ckpt files found in {ckpt_dir}")
        sys.exit(1)

    log.info(f"Found {len(checkpoints)} checkpoints in {ckpt_dir}")
    log.info(f"Output directory: {output_dir}")

    # Build base command from CLI args (excluding model_weights_path and output_path)
    base_cmd = [sys.executable, "src/predict.py"]
    for arg in sys.argv[1:]:
        key = arg.split("=", 1)[0] if "=" in arg else None
        if key not in ("model_weights_path", "output_path"):
            base_cmd.append(arg)

    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.stem
        output_path = f"{output_dir}/{ckpt_name}.pkl"

        if Path(output_path).exists():
            log.info(f"Skipping {ckpt_name} - output already exists: {output_path}")
            continue

        log.info(f"Processing checkpoint: {ckpt_name}")

        # Escape = for Hydra
        ckpt_path_escaped = str(ckpt_path).replace("=", "\\=")
        output_path_escaped = output_path.replace("=", "\\=")

        cmd = base_cmd + [
            f"model_weights_path={ckpt_path_escaped}",
            f"output_path={output_path_escaped}",
        ]

        log.info(f"Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, cwd=original_cwd, check=True)
            log.info(f"Completed: {ckpt_name}")
        except subprocess.CalledProcessError as e:
            log.error(f"Failed processing {ckpt_name}: {e}")


if __name__ == "__main__":
    main()
