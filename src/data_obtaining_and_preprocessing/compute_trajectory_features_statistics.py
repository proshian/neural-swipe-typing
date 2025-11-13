import argparse
import json
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset import SwipeDataset
from ns_tokenizers import CharLevelTokenizerv2
from feature_extraction.swipe_feature_extractors import TrajectoryFeatureExtractor
from feature_extraction.normalizers import identity_function

STAT_KEYS = ["dt", "velocity_x", "velocity_y", "acceleration_x", "acceleration_y"]


class OnlineMeanStd:
    """
    Online mean and standard deviation calculator.

    Given a tensor updates the mean and variance using parallel
    algorithm (a generalization of Welford's algorithm).
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from the mean

    def update(self, x: torch.Tensor):
        x = x.flatten().double()
        m = x.numel()
        if m == 0:
            return

        mean_batch = x.mean().item()
        M2_batch = ((x - mean_batch)**2).sum().item()

        if self.n == 0:
            self.n = m
            self.mean = mean_batch
            self.M2 = M2_batch
        else:
            delta = mean_batch - self.mean
            new_n = self.n + m
            self.mean = (self.n * self.mean + m * mean_batch) / new_n
            self.M2 += M2_batch + (delta ** 2) * (self.n * m) / new_n
            self.n = new_n

    def get_statistics(self):
        variance = self.M2 / (self.n) if self.n > 1 else 0.0
        return {"mean": self.mean, "std": variance ** 0.5}



def collate_for_statistics(batch):
    dt_list = []
    dxdt_list = []
    dydt_list = []
    d2xdt2_list = []
    d2ydt2_list = []

    for ((features, _), _) in batch:
        traj_feats = features[0]  # (seq_len, num_feats)
        dt_list.append(traj_feats[:, 2][1:])
        dxdt_list.append(traj_feats[:, 3][1:-1])
        dydt_list.append(traj_feats[:, 4][1:-1])
        d2xdt2_list.append(traj_feats[:, 5][1:-1])
        d2ydt2_list.append(traj_feats[:, 6][1:-1])

    return {
        "dt": torch.cat(dt_list, dim=0),
        "velocity_x": torch.cat(dxdt_list, dim=0),
        "velocity_y": torch.cat(dydt_list, dim=0),
        "acceleration_x": torch.cat(d2xdt2_list, dim=0),
        "acceleration_y": torch.cat(d2ydt2_list, dim=0),
    }



def make_trajectory_extractor():
    return TrajectoryFeatureExtractor(
        include_dt=True,
        include_velocities=True,
        include_accelerations=True,
        x_normalizer = identity_function,
        y_normalizer = identity_function,
        dt_normalizer = identity_function,
        velocity_x_normalizer = identity_function,
        velocity_y_normalizer = identity_function,
        acceleration_x_normalizer = identity_function,
        acceleration_y_normalizer = identity_function
    )


def compute_statistics(data_path, voc_path, output_json, 
                       num_workers, batch_size, total) -> None:
    tokenizer = CharLevelTokenizerv2(voc_path)
    feature_extractor = make_trajectory_extractor()
    dataset = SwipeDataset(
        data_path=data_path,
        word_tokenizer=tokenizer,
        grid_name_to_swipe_feature_extractor=defaultdict(lambda: feature_extractor),
        total=total
    )

    stats = {key: OnlineMeanStd() for key in STAT_KEYS}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_for_statistics
    )

    for batch_features in tqdm(dataloader, desc="Computing statistics"):
        for key in STAT_KEYS:
            stats[key].update(batch_features[key])

    output = {k: stats[k].get_statistics() for k in STAT_KEYS}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--voc_path", required=True)  # Only to create tokenizer
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=300)
    args = parser.parse_args()

    compute_statistics(
        args.train_data_path,
        args.voc_path,
        args.output_json,
        args.num_workers,
        args.batch_size,
        args.total,
    )


if __name__ == "__main__":
    main()
