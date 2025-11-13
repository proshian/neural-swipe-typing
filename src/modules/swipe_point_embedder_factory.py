from typing import Union

import torch

from modules.swipe_point_embedders import (NearestEmbeddingWithPos,
                                           SeparateTrajAndWEightedEmbeddingWithPos,
                                           SeparateTrajAndTrainableWeightedEmbeddingWithPos,
                                           SeparateTrajAndNearestEmbeddingWithPos)
from modules.spe_factory_utils import get_kb_centers_tensor
from feature_extraction.normalizers import MinMaxNormalizer


def swipe_point_embedder_factory(
    config: dict,
    device: Union[torch.device, str]
) -> torch.nn.Module:
    """
    Factory function to create a swipe point embedder based on the configuration.
    
    Parameters:
    -----------
    config: dict
        Configuration dictionary containing parameters for the embedder.
    
    Returns:
    --------
    nn.Module
        An instance of a swipe point embedder.
    """
    params = config['params']
    
    if config['type'] == 'nearest':
        return NearestEmbeddingWithPos(
            params['n_keys'],
            params['key_emb_size'],
            params['max_len'],
            device,
            params['dropout']
        )
    
    elif config['type'] == 'separate_traj_and_weighted':
        return SeparateTrajAndWEightedEmbeddingWithPos(
            params['n_keys'],
            params['key_emb_size'],
            params['max_len'],
            device,
            params['dropout']
        )
    
    elif config['type'] == 'separate_traj_and_nearest':
        return SeparateTrajAndNearestEmbeddingWithPos(
            params['n_keys'],
            params['key_emb_size'],
            params['max_len'],
            device,
            params['dropout']
        )
    
    elif config['type'] == 'separate_traj_and_trainable_weighted':

        kb_x_normalizer = MinMaxNormalizer(
            params['kb_x_min'],
            params['kb_x_max']
        )
        kb_y_normalizer = MinMaxNormalizer(
            params['kb_y_min'],
            params['kb_y_max']
        )
        
        key_centers = get_kb_centers_tensor(
            params['grid_path'],
            params['grid_name'],
            params['kb_tokenizer_json_path'],
            kb_x_normalizer,
            kb_y_normalizer
        )
        
        return SeparateTrajAndTrainableWeightedEmbeddingWithPos(
            params['n_keys'],
            params['key_emb_size'],
            params['max_len'],
            device,
            params['dropout'],
            key_centers
        )
    
    else:
        raise ValueError(f"Unknown swipe point embedder type: {config['type']}")
