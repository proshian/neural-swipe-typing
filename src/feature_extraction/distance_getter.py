from typing import Tuple

import torch
from torch import Tensor

from grid_processing_utils import get_kb_label
from ns_tokenizers import KeyboardTokenizer


def compute_pairwise_distances(dots: Tensor, centers: Tensor) -> Tensor:
    """
    Arguments:
    ----------
    dots: Tensor
        Dots tensor. dots.shape = (*DOT_DIMS, 2). DOT_DIMS: tuple = (S1, S2, S3, ... SD).
    centers: Tensor
        Centers tensor. centers.shape = (K, 2). K is number of centers.

    Returns:
    --------
    Tensor
        Distance tensor. Distance tensor.shape = (*DOT_DIMS, K).
        euclidean distance is used.
    
    Example:
    --------
    dots = torch.tensor([[1, 2], [3, 4], [5, 6]])
    centers = torch.tensor([[1, 2], [3, 4]])
    distance(dots, centers) -> torch.tensor([[0, 8], [8, 0], [32, 8]])
    """
    # (*DOT_DIMS, 1, 2)
    dots_exp = dots.unsqueeze(-2)
    # (K, 2) -> (1, 1, ..., 1, K, 2)
    centers_exp = centers.view(*([1] * (dots.dim() - 1)), *centers.shape) 
    return torch.sqrt(torch.pow((centers_exp - dots_exp), 2).sum(dim=-1))


class DistanceGetter(torch.nn.Module):
    """
    Computes distances from coordinates to key centers.
    Handles missing keys via masking.
    """

    def __init__(self,
                 grid: dict,
                 tokenizer: KeyboardTokenizer,
                 missing_distance_val: float = float('inf')) -> None:
        """
        Arguments:
        ----------
        grid: dict
        tokenizer: KeyboardTokenizer
        missing_distance_val: float
            Value to fill for distances to keys that are not present in the grid.
            Defaults to +inf.
        """
        super().__init__()
        self.missing_distance_val = missing_distance_val
        centers, mask = self._get_centers(grid, tokenizer)
        self.register_buffer('centers', centers)
        self.register_buffer('mask', mask)

    def _get_centers(self, grid: dict, 
                     tokenizer: KeyboardTokenizer
                     ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
        --------
        centers: 
            Tensor of shape (vocab_size, 2)
        mask: 
            BoolTensor of shape (vocab_size,) — True where center is missing
        """
        known_non_special_token_ids = tokenizer.get_all_non_special_token_ids()
        
        centers = torch.empty((len(tokenizer), 2),
                             dtype=torch.float32)

        present_tokens = set()

        for key in grid['keys']:
            label = get_kb_label(key)
            token = tokenizer.get_token(label)
            if token in known_non_special_token_ids:
                hb = key['hitbox']
                centers[token] = torch.tensor(
                    [hb['x'] + hb['w'] / 2, hb['y'] + hb['h'] / 2],
                )
                present_tokens.add(token)

        mask = torch.ones((len(tokenizer),), dtype=torch.bool)
        mask[torch.tensor(list(present_tokens))] = False

        return centers, mask

    def forward(self, coords: Tensor) -> Tensor:
        """
        Arguments:
        ----------
        coords: Tensor
            Coordinates tensor of shape (..., 2)

        Returns:
        --------
        distances: Tensor
            Tensor of shape (..., K), 
            where K is the (max token id + 1) among key_labels_of_interest.
        """
        coords = coords.to(dtype=torch.float32)
        dists = compute_pairwise_distances(coords, self.centers)  # (..., K)
        dists[..., self.mask] = self.missing_distance_val
        return dists
