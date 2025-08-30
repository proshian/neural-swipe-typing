from typing import Dict, Any
import argparse
import os
import logging
from typing import Optional

import torch


logger = logging.getLogger(__name__)


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def ckpt_to_torch_state(ckpt: Dict[str, Any], model_prefix: str = 'model.'):
    return {remove_prefix(k, model_prefix): v for k, v in ckpt['state_dict'].items()}


######################################################################################


def convert_and_save_dir(ckpt_root: str, out_root: str, 
                         device: torch.device) -> None:
    """
    Given a nested directory of .ckpt files, converts each to .pt 
    and saves them to out_root preserving the directory structure.
    """
    for root, dirs, files in os.walk(ckpt_root):
        for file in files:
            if not file.endswith('.ckpt'):
                continue

            ckpt_path = os.path.join(root, file)
            orig_path_no_ext, orig_ext = os.path.splitext(ckpt_path)
            assert orig_ext == '.ckpt'
            orig_path_no_ext_no_root = os.path.relpath(orig_path_no_ext, ckpt_root)
            out_path = os.path.join(out_root, orig_path_no_ext_no_root + '.pt')
            if os.path.exists(out_path):
                logger.info(f"Skipping {ckpt_path} as {out_path} already exists")
                continue
            out_parent = os.path.dirname(out_path)
            if not os.path.exists(out_parent):
                os.makedirs(out_parent)
            # print(f"Converting {ckpt_path} to {out_path}")
            convert_and_save_file(ckpt_path, out_path, device)


def convert_and_save_file(ckpt_path: str, 
                          out_path: str, 
                          device: torch.device, 
                          make_dir_if_absent: bool = False) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt_to_torch_state(ckpt)
    if make_dir_if_absent and not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    torch.save(state_dict, out_path)


def setup_logging(log_file: Optional[str] = None) -> None:
    if log_file is not None:
        handlers = [logging.StreamHandler(), logging.FileHandler(log_file, mode='w')]
    else:
        handlers = None
    logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s', handlers=handlers)
    


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    setup_logging()

    assert os.path.exists(args.ckpt_path), f"ckpt_path does not exist: {args.ckpt_path}"

    device = torch.device(args.device)

    if os.path.isfile(args.ckpt_path):
        assert os.path.exists(args.ckpt_path), f"ckpt_path does not exist: {args.ckpt_path}"
        convert_and_save_file(args.ckpt_path, args.out_path, device)
    else:
        convert_and_save_dir(args.ckpt_path, args.out_path, device)
