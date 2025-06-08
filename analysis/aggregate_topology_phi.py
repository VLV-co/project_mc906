import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict

# Constants for each topology
topologies = {
    'hex': {
        'cells_number': 19,
        'possible_moves': 6,
        'connection_degree': 4.11,
        'data_dir': 'data/hex'
    },
    'square': {
        'cells_number': 16,
        'possible_moves': 4,
        'connection_degree': 3,
        'data_dir': 'data/square'
    },
    'triangle': {
        'cells_number': 10,
        'possible_moves': 6,
        'connection_degree': 3,
        'data_dir': 'data/triangle'
    }
}

MU: float = 0.05


def collect_max_tiles(data_dir: str) -> List[int]:
    """Collect all max_tile values from all CSVs in a directory."""
    max_tiles = []
    for csv_file in glob.glob(os.path.join(data_dir, '*.csv')):
        df = pd.read_csv(csv_file)
        if 'max_tile' in df.columns:
            max_tiles.extend(df['max_tile'].tolist())
    return max_tiles


def compute_phi(max_tiles: List[int], mu: float = MU) -> float:
    """Compute phi as specified."""
    if not max_tiles:
        return float('nan')
    log2_tiles = np.log2(max_tiles)
    mean_log2 = np.mean(log2_tiles)
    sigma = np.std(log2_tiles)
    phi = mean_log2 * (1 + mu * sigma)
    return phi


def main():
    rows = []
    for topo, params in topologies.items():
        max_tiles = collect_max_tiles(params['data_dir'])
        phi = compute_phi(max_tiles)
        rows.append({
            'cells_number': params['cells_number'],
            'possible_moves': params['possible_moves'],
            'connection_degree': params['connection_degree'],
            'phi': phi
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv('data/aggregated/topologies_phi.csv', index=False)


if __name__ == '__main__':
    main() 