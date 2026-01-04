from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .distance_getter import DistanceGetter
from grid_processing_utils import get_avg_half_key_diag
from ns_tokenizers import KeyboardTokenizer


class KeyWeightsGetter(nn.Module):
    def __init__(self,
                 grid: dict,
                 tokenizer: KeyboardTokenizer,
                 weights_function: Callable,
                 missing_value_weight: float = 0.0,
                ) -> None:
        super().__init__()
        key_labels_of_interest = tokenizer.get_all_non_special_tokens()
        self.distance_getter = DistanceGetter(grid, tokenizer)
        self.half_key_diag = get_avg_half_key_diag(grid, key_labels_of_interest)
        self.missing_value_weight = missing_value_weight
        self.weights_function = weights_function

    def forward(self, coords: Tensor) -> Tensor:
        distances = self.distance_getter(coords)
        mask = self.distance_getter.mask
        present_distances = distances[..., ~mask]
        present_distances_scaled = present_distances / self.half_key_diag
        present_weights = self.weights_function(present_distances_scaled)
        weights = torch.full_like(distances, self.missing_value_weight)
        weights[..., ~mask] = present_weights
        return weights
