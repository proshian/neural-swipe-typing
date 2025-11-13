from typing import Callable

import torch

from grid_processing_utils import get_grid, get_kb_key_center
from ns_tokenizers import KeyboardTokenizer

def get_id_to_kb_centers_list(kb_tokenizer, keys):
    kb_centers_dict = dict()
    for key in keys:
        if 'label' not in key:
            continue
        if key['label'] not in kb_tokenizer.t2i:
            continue
        key_id = kb_tokenizer.t2i[key['label']]
        kb_centers_dict[key_id] = get_kb_key_center(key['hitbox'])
    
    for t, i in kb_tokenizer.t2i.items():
        if i not in kb_centers_dict:
            kb_centers_dict[i] = (-1, -1)

    return kb_centers_dict
    


def dict_to_sorted_list(d):
    return [v for k, v in sorted(d.items())]

def get_kb_centers_tensor(grid_path: str,
                          grid_name: str,
                          kb_tokenizer_json: str,
                          kb_x_scaler: Callable, 
                          kb_y_scaler: Callable):
    grid = get_grid(grid_name, grid_path)
    keys = grid['keys']


    kb_tokenizer = KeyboardTokenizer(kb_tokenizer_json)
    
    # legacy thing
    assert len(kb_tokenizer.t2i) == len(kb_tokenizer.i2t)
    if len(kb_tokenizer.t2i) != 37:
        kb_tokenizer.i2t.append('<extra>')
        kb_tokenizer.t2i['<extra>'] = len(kb_tokenizer.i2t) - 1
    assert len(kb_tokenizer.t2i) == 37


    kb_centers_dict = get_id_to_kb_centers_list(kb_tokenizer, keys)
    kb_centers_sorted_lst = dict_to_sorted_list(kb_centers_dict)
    kb_centers_tensor = torch.tensor(kb_centers_sorted_lst).float()
    mask = kb_centers_tensor == -1
    kb_centers_tensor[:, 0].apply_(kb_x_scaler)
    kb_centers_tensor[:, 1].apply_(kb_y_scaler)
    kb_centers_tensor = kb_centers_tensor.masked_fill(mask, -1)
