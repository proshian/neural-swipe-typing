from typing import List, Dict

import torch
from torch import Tensor
import torch.nn as nn

from src.feature_extraction.swipe_feature_extractors import SwipeFeatureExtractor


class ModelWithFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, grid_name_to_feature_extractor: Dict[str, SwipeFeatureExtractor]):
        super().__init__()
        self.model = model
        self.extractors = nn.ModuleDict(grid_name_to_feature_extractor)

    def _get_features(self, x: Tensor, y: Tensor, t: Tensor, grid_names: List[str]) -> List[Tensor]:
        """
        Arguments:
        ----------
        x: Tensor (seq_len, batch_size)
        y: Tensor (seq_len, batch_size)
        t: Tensor (seq_len, batch_size)
        grid_names: list of grid names of length batch_size
        """
        grid_to_idx = self._get_grid_to_indices(grid_names)

        if len(grid_to_idx) == 1:
            feats = self._extract_features_single_grid(x, y, t, next(iter(grid_to_idx)))
        else:
            feats = self._extract_features_multi_grid(x, y, t, grid_to_idx)
        return feats
    
    def _get_grid_to_indices(self, grid_names: List[str]) -> Dict[str, List[int]]:
        grid_to_idx = {}
        for i, name in enumerate(grid_names):
            grid_to_idx.setdefault(name, []).append(i)
        return grid_to_idx

    def _extract_features_single_grid(self, x: Tensor, y: Tensor, t: Tensor, grid_name: str):
        if grid_name not in self.extractors:
            raise ValueError(f"Feature extractor for grid '{grid_name}' not found.")
        return self.extractors[grid_name](x, y, t)

    def _extract_features_multi_grid(self, x: Tensor, y: Tensor, t: Tensor, grid_to_idx: Dict[str, List[int]]):
        device = x.device
        feature_parts = []
        indices_parts = []
        
        for name, idxs in grid_to_idx.items():
            if name not in self.extractors:
                raise ValueError(f"Feature extractor for grid '{name}' not found.")
            
            # Create index tensor
            idxs_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
            
            # Assuming shape (seq_len, batch_size)
            sub_x = x.index_select(1, idxs_tensor)
            sub_y = y.index_select(1, idxs_tensor)
            sub_t = t.index_select(1, idxs_tensor)
            
            # Run extractor
            feats = self.extractors[name](sub_x, sub_y, sub_t)
            
            feature_parts.append(feats)
            indices_parts.append(idxs_tensor)
        
        all_indices = torch.cat(indices_parts)
        # Permutation that sorts all_indices back to 0..N-1
        perm = torch.argsort(all_indices)
        
        merged_features = []
        num_feature_types = len(feature_parts[0])
        
        for i in range(num_feature_types):            
            feat_list_for_attr = [fp[i] for fp in feature_parts]
            feat_cat = torch.cat(feat_list_for_attr, dim=1)
            
            # Reorder to match original batch order
            feat_ordered = feat_cat.index_select(1, perm)
            merged_features.append(feat_ordered)
            
        return merged_features

    def forward(self, x: Tensor, y: Tensor, t: Tensor, grid_names: List[str], *args, **kwargs):
        """
        Args:
            x: (seq_len, batch_size)
            y: (seq_len, batch_size)
            t: (seq_len, batch_size)
            grid_names: list of grid names of length batch_size
        """
        feats = self._get_features(x, y, t, grid_names)
        return self.model(feats, *args, **kwargs)
