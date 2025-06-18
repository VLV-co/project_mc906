import os
import glob
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple

# Constants for each topology
# Agora sem cells_number fixo, pois pode variar conforme o sufixo do diretório

topologies = {
    'hex': {
        'possible_moves': 6,
        'connection_degree': 84/19,
        'data_dirs': [
            ('data/hex', 19),
            ('data/hex_2', 7),
            ('data/hex_3', 9)
        ]
    },
    'square': {
        'possible_moves': 4,
        'connection_degree': 3,
        'data_dirs': [
            ('data/square', 16),
            ('data/square_3', 9)
        ]
    },
    'triangle': {
        'possible_moves': 6,
        'connection_degree': 3.6,
        'data_dirs': [
            ('data/triangle', 10)
        ]
    }
}

MU: float = 0.05


def extract_depth_from_filename(filename: str) -> int:
    """Extract depth from filename, return 0 if not found (e.g., for random)."""
    match = re.search(r'depth_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def collect_max_tiles_by_depth(data_dir: str) -> Dict[int, List[int]]:
    """Collect all max_tile values from all CSVs in a directory, grouped by depth."""
    max_tiles_by_depth: Dict[int, List[int]] = {}
    for csv_file in glob.glob(os.path.join(data_dir, '*.csv')):
        depth = extract_depth_from_filename(os.path.basename(csv_file))
        df = pd.read_csv(csv_file)
        if 'max_tile' in df.columns:
            if depth not in max_tiles_by_depth:
                max_tiles_by_depth[depth] = []
            max_tiles_by_depth[depth].extend(df['max_tile'].tolist())
    return max_tiles_by_depth


def main():
    rows = []
    for topo, params in topologies.items():
        for data_dir, cells_number in params['data_dirs']:
            if not os.path.exists(data_dir):
                continue
            max_tiles_by_depth = collect_max_tiles_by_depth(data_dir)
            for depth, max_tiles in max_tiles_by_depth.items():
                if not max_tiles:
                    continue
                mean_max_tile = float(np.mean(np.log2(max_tiles)))
                rows.append({
                    'topology': topo,
                    'cells_number': cells_number,
                    'possible_moves': params['possible_moves'],
                    'connection_degree': params['connection_degree'],
                    'depth': depth,
                    'mean_log2_max_tile': mean_max_tile
                })
    out_df = pd.DataFrame(rows)
    out_df.to_csv('data/aggregated/experiments.csv', index=False)


if __name__ == '__main__':
    main() 