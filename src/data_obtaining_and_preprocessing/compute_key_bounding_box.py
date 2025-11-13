import argparse
import json
from typing import Dict, Set

from grid_processing_utils import get_kb_label


def compute_bounding_box(grid: Dict, labels_of_interest: Set[str]) -> Dict:
    """
    Computes the bounding box for keys in the grid with labels in labels_of_interest.

    Args:
        grid: A dictionary representing the keyboard grid.
        labels_of_interest: A set of key labels to consider.

    Returns:
        A dictionary with 'x_min', 'y_min', 'x_max', 'y_max' representing the bounding box,
        or None if no matching keys are found.
    """
    x_min = float('inf')
    y_min = float('inf')
    x_max = -float('inf')
    y_max = -float('inf')
    
    found = False
    for key in grid['keys']:
        label = get_kb_label(key)
        
        if label in labels_of_interest:
            found = True
            hb = key['hitbox']
            x = hb['x']
            y = hb['y']
            w = hb['w']
            h = hb['h']
            current_x_max = x + w
            current_y_max = y + h
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, current_x_max)
            y_max = max(y_max, current_y_max)
    
    if not found:
        return None
    
    return {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute bounding boxes around specified key labels in keyboard grids.')
    parser.add_argument('--grids_path', required=True,
                       help='Path to JSON file containing grid definitions.')
    parser.add_argument('--labels', nargs='+', required=True,
                       help='Space-separated list of key labels of interest (allowed characters).')
    parser.add_argument('--output_json', required=True,
                       help='Path to save the output JSON file with bounding boxes.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    labels_of_interest = set(args.labels)

    with open(args.grids_path, 'r', encoding='utf-8') as f:
        grids = json.load(f)

    output = {}
    for grid_name, grid in grids.items():
        bbox = compute_bounding_box(grid, labels_of_interest)
        if bbox is None:
            raise ValueError(f"No keys of interest were present in the grid {grid_name}")
        output[grid_name] = bbox
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
