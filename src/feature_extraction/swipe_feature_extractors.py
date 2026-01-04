from typing import List, Optional
from collections.abc import Callable
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn

from .normalizers import identity_function


class SwipeFeatureExtractor(nn.Module, ABC):
    """
    Abstract base class for swipe feature extractors.
    A SwipeFeatureExtractor is any nn.Module that takes in x, y, t tensors
    and outputs a list of feature tensors.
    """
    @abstractmethod
    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        """
        Extract features from raw coordinates and times.
        All input tensors can have shape (seq_len,) or (seq_len, batch_size, ...).
        """
        pass


class MultiFeatureExtractor(SwipeFeatureExtractor):
    """
    Extracts multiple feature tensors via a list of feature extractors it holds.
    """
    def __init__(self, extractors: List[SwipeFeatureExtractor]) -> None:
        super().__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        aggregated_features: List[Tensor] = []
        for extractor in self.extractors:
            aggregated_features.extend(extractor(x, y, t))
        return aggregated_features


class TrajectoryFeatureExtractor(SwipeFeatureExtractor):
    """
    Extracts trajectory features such as x, y, dt and coordinate derivatives.

    Expects x, y and t to have seq_len as the 0-th dimension. 
    May have only one dimension, i.e. (seq_len, ) or more, e.g. (seq_len, batch_size).
    """
    def __init__(self,
                 include_dt: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 x_normalizer: Callable[[Tensor], Tensor],
                 y_normalizer: Callable[[Tensor], Tensor],
                 dt_normalizer: Callable[[Tensor], Tensor] = identity_function,
                 velocity_x_normalizer: Callable[[Tensor], Tensor] = identity_function,
                 velocity_y_normalizer: Callable[[Tensor], Tensor] = identity_function,
                 acceleration_x_normalizer: Callable[[Tensor], Tensor] = identity_function,
                 acceleration_y_normalizer: Callable[[Tensor], Tensor] = identity_function,
                 ) -> None:
        """
        Arguments:
        ----------
        include_dt: bool
            If True, includes time since prev point (dt) as a feature.
        include_velocities : bool
            If True, includes dx/dt and dy/dt as features.
        include_accelerations : bool
            If True, includes d²x/dt² and d²y/dt² as features.
        coordinate_normalizer : Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
            Callable to normalize x and y coordinates.
        dt_normalizer : Callable[[Tensor], Tensor], optional
            Callable to normalize dt. Defaults to identity function.
        velocities_normalizer : Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]], optional
            Callable to normalize dx/dt and dy/dt. Defaults to identity function.
        accelerations_normalizer : Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]], optional
            Callable to normalize d²x/dt² and d²y/dt². Defaults to identity function.
        """
        super().__init__()
        self.include_dt = include_dt
        self.include_velocities = include_velocities
        self.include_accelerations = include_accelerations
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.dt_normalizer = dt_normalizer
        self.velocity_x_normalizer = velocity_x_normalizer
        self.velocity_y_normalizer = velocity_y_normalizer
        self.acceleration_x_normalizer = acceleration_x_normalizer
        self.acceleration_y_normalizer = acceleration_y_normalizer

    def _get_central_difference_derivative(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Calculates dx/dt for a list of x coordinates and a list of t coordinates.

        Arguments:
        ----------
        X : Tensor of shape (seq_len,) or (seq_len, ...)
            x (position) coordinates.
        T : Tensor of shape (seq_len,) or (seq_len, ...)
            T[i] = time (ms) from swipe start corresponding to X[i].

        Example:
        --------
        x0 x1 x2 x3
        t0 t1 t2 t3
        dx_dt = [0, (x2 - x0)/(t2 - t0), (x3 - x1)/(t3 - t1), 0]
        """
        dx_dt = torch.zeros_like(x)
        numerator = x[2:] - x[:-2]
        denominator = t[2:] - t[:-2]
        
        # Avoid division by zero (happens if `t` is padded with zeros during collation).
        # This is probably overly cautious, as these positions
        # will be masked later.
        denominator = torch.where(denominator == 0, torch.full_like(denominator, 1e-6), denominator)

        dx_dt[1:-1] = numerator / denominator
        return dx_dt

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        """
        Returns:
        --------
        List[Tensor]: 
            A list containing a single tensor of shape (seq_len, num_features)
            containing the trajectory features.
        """
        x_norm, y_norm = self.x_normalizer(x), self.y_normalizer(y)

        traj_feats_lst = [x_norm, y_norm]

        if self.include_dt:
            dt_from_prev  = torch.zeros_like(t)
            dt_from_prev[1:] = self.dt_normalizer(t[1:] - t[:-1])
            traj_feats_lst.append(dt_from_prev)

        is_velocities_needed = (self.include_velocities or self.include_accelerations)

        if is_velocities_needed:
            dx_dt = self._get_central_difference_derivative(x, t)
            dy_dt = self._get_central_difference_derivative(y, t)

        if self.include_velocities:
            traj_feats_lst.extend([
                self.velocity_x_normalizer(dx_dt),
                self.velocity_y_normalizer(dy_dt)
            ])

        if self.include_accelerations:
            d2x_dt2 = self._get_central_difference_derivative(dx_dt, t)
            d2y_dt2 = self._get_central_difference_derivative(dy_dt, t)
            traj_feats_lst.extend([
                self.acceleration_x_normalizer(d2x_dt2),
                self.acceleration_y_normalizer(d2y_dt2)
            ])
        
        traj_feats = torch.stack(traj_feats_lst, dim=-1)

        return [traj_feats]


class CoordinateFunctionFeatureExtractor(SwipeFeatureExtractor):
    def __init__(self,
                 value_fn: Callable[[Tensor], Tensor],
                 cast_dtype: Optional[torch.dtype] = None
                 ) -> None:
        """
        An adapter to make callables that accept stack(x, y) 
        satisfy the SwipeFeatureExtractor interface.

        Arguments:
        ----------
        value_fn: Callable[[Tensor], Tensor]
            Function accepting Tensor (N, 2) of (x, y)
            and returning (N, feature_size).
        cast_dtype: Optional[torch.dtype]:
            Dtype that x and y are casted to before applying value_fn.
            Primer use: cast to integer if value_fn is an instance of GridLookup.
        """
        super().__init__()
        self.value_fn = value_fn
        self.cast_dtype = cast_dtype

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        coords = torch.stack([x, y], dim=-1)
        
        if self.cast_dtype is not None:
            coords = coords.to(dtype=self.cast_dtype)
        features = self.value_fn(coords)
        return [features]
