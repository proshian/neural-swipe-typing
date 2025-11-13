from collections.abc import Callable
from typing import Tuple

import torch
from torch import Tensor

_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}

def is_integer_tensor(tensor: Tensor) -> bool:
    return tensor.dtype in _INTEGER_DTYPES

class GridLookup:
    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        value_fn: Callable[[Tensor], Tensor],
    ):
        """
        Arguments:
        ----------
        grid_width: int
            Keyboard grid width.
        grid_height: int
            Keyboard grid height.
        value_fn: Callable[[Tensor], Tensor]
            Function accepting Tensor (N, 2) of (x, y)
            and returning either (N,) or (N, feature_size).
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.value_fn = value_fn
        self.feature_size, self.output_has_feature_dim = self._get_feature_properties()
        self.lookup_tensor = self._build_lookup_tensor()

    def _get_feature_properties(self) -> Tuple[int, bool]:
        sample_coords = torch.zeros((1, 2), dtype=torch.int32)
        features = self.value_fn(sample_coords)
        if features.dim() == 1:
            return 1, False
        return features.shape[1], True
    
    def _build_lookup_tensor(self) -> Tensor:
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid_width),
            torch.arange(self.grid_height),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)  # shape (W*H, 2)

        features = self.value_fn(coords)
        if not self.output_has_feature_dim:
            features = features.unsqueeze(-1)
            
        assert features.shape == (coords.size(0), self.feature_size), \
            f"Expected ({coords.size(0)}, {self.feature_size}), got {features.shape}"

        return features.view(self.grid_width, self.grid_height, self.feature_size)

    def __call__(self, coords: Tensor) -> Tensor:
        if not is_integer_tensor(coords):
            raise ValueError("coords must be an integer tensor, " \
                             f"got {coords.dtype}")
        
        x, y = coords.unbind(dim=-1)

        in_bounds = (x >= 0) & (x < self.grid_width) & (y >= 0) & (y < self.grid_height)
        output = torch.empty((x.size(0), self.feature_size), dtype=self.lookup_tensor.dtype)

        if in_bounds.any():
            x_ib = x[in_bounds]
            y_ib = y[in_bounds]
            output[in_bounds] = self.lookup_tensor[x_ib, y_ib]

        if (~in_bounds).any():
            coords_oob = torch.stack((x[~in_bounds], y[~in_bounds]), dim=1)
            values_oob = self.value_fn(coords_oob)
            if not self.output_has_feature_dim:
                values_oob.unsqueeze_(-1)
            output[~in_bounds] = values_oob

        if not self.output_has_feature_dim:
            output = output.squeeze(-1)

        return output
