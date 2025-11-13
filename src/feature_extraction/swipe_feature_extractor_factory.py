from typing import List

import torch

from feature_extraction.swipe_feature_extractors import (
    MultiFeatureExtractor,
    TrajectoryFeatureExtractor,
    CoordinateFunctionFeatureExtractor,
)
from ns_tokenizers import KeyboardTokenizer
from feature_extraction.nearest_key_getter import NearestKeyGetter
from feature_extraction.key_weights_getter import KeyWeightsGetter
from feature_extraction.normalizers import (
    MinMaxNormalizer,
    MeanStdNormalizer,
)
from feature_extraction.grid_lookup import GridLookup
from feature_extraction.key_weights_functions import (
    WeightsFnV1,
    WeightsFunctionV1Softmax,  
    WeightsFunctionSigmoidNormalizedV1
)


KEY_WEIGHTS_FUNCTIONS = {
    "v1": WeightsFnV1,
    "v1_softmax": WeightsFunctionV1Softmax,
    "sigmoid_normalized_v1": WeightsFunctionSigmoidNormalizedV1,
}


def swipe_feature_extractor_factory(grid: dict,
                                    keyboard_tokenizer: KeyboardTokenizer,
                                    trajectory_features_statistics: dict,
                                    bounding_boxes: dict,
                                    grid_name: str,
                                    component_configs: List[dict]):
    extractors = []
    for component_config in component_configs:
        extractor_type = component_config["type"]
        extractor_params = component_config["params"]

        if extractor_type == "trajectory":
            extractors.append(
                TrajectoryFeatureExtractor(
                    include_dt=extractor_params["include_dt"],
                    include_velocities=extractor_params["include_velocities"],
                    include_accelerations=extractor_params["include_accelerations"],
                    x_normalizer=MinMaxNormalizer(
                        bounding_boxes[grid_name]["x_min"],
                        bounding_boxes[grid_name]["x_max"]
                    ),
                    y_normalizer=MinMaxNormalizer(
                        bounding_boxes[grid_name]["y_min"],
                        bounding_boxes[grid_name]["y_max"]
                    ),
                    dt_normalizer=MeanStdNormalizer(
                        **trajectory_features_statistics["dt"]
                    ),
                    velocity_x_normalizer=MeanStdNormalizer(
                        **trajectory_features_statistics["velocity_x"]
                    ),
                    velocity_y_normalizer=MeanStdNormalizer(
                        **trajectory_features_statistics["velocity_y"]
                    ),
                    acceleration_x_normalizer=MeanStdNormalizer(
                        **trajectory_features_statistics["acceleration_x"]
                    ),
                    acceleration_y_normalizer=MeanStdNormalizer(
                        **trajectory_features_statistics["acceleration_y"]
                    )
                )
            )   
        elif extractor_type == "nearest_key":
            value_fn = NearestKeyGetter(
                grid=grid,
                tokenizer=keyboard_tokenizer,
            )
            cast_dtype = None

            if extractor_params.get("use_lookup", False):
                value_fn = GridLookup(grid["width"], grid["height"], value_fn)
                cast_dtype=torch.int32
            
            extractors.append(
                CoordinateFunctionFeatureExtractor(value_fn, cast_dtype)
            )
        elif extractor_type == "key_weights":
            weights_function_ctor = KEY_WEIGHTS_FUNCTIONS[extractor_params["weights_function"]["function_name"]]
            weights_function = weights_function_ctor(
                **extractor_params["weights_function"]["params"]
            )
            value_fn = KeyWeightsGetter(
                grid=grid,
                tokenizer=keyboard_tokenizer,
                weights_function=weights_function,
                missing_value_weight=extractor_params.get("missing_value_weight", 0.0),
            )
            cast_dtype = None

            if extractor_params.get("use_lookup", False):
                value_fn = GridLookup(grid["width"], grid["height"], value_fn)
                cast_dtype=torch.int32
            
            extractors.append(
                CoordinateFunctionFeatureExtractor(value_fn, cast_dtype)
            )
        else:
            raise ValueError(f"Unknown feature extractor type: {extractor_type}")
    return MultiFeatureExtractor(extractors)
