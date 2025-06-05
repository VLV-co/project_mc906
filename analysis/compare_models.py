import os
import re
from typing import Dict, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Regex to extract heuristic, depth, and size from filename
# Handles expectimax_heuristic_depth_X_board_size_Y.csv and random_board_size_Y.csv
def parse_filename(filename: str) -> Tuple[str, int, int]:
    match = re.match(r"expectimax_([a-z_]+)_depth_(\d+)_([a-z]+)_size_(\d+)\.csv", filename)
    if match:
        heuristic, depth, board_type, size = match.groups()
        return heuristic, int(depth), int(size)
    match = re.match(r"random_([a-z]+)_size_(\d+)\.csv", filename)
    if match:
        board_type, size = match.groups()
        return 'random', 0, int(size)
    raise ValueError(f"Unrecognized filename: {filename}")

def load_datasets(data_dir: str) -> Dict[str, Dict[str, Dict[int, pd.DataFrame]]]:
    """
    Loads datasets into a nested dict: datasets[board_type][heuristic][depth] = DataFrame
    """
    datasets: Dict[str, Dict[str, Dict[int, pd.DataFrame]]] = {}
    for board_type in os.listdir(data_dir):
        board_path = os.path.join(data_dir, board_type)
        if not os.path.isdir(board_path):
            continue
        for fname in os.listdir(board_path):
            if not fname.endswith('.csv'):
                continue
            try:
                heuristic, depth, size = parse_filename(fname)
            except ValueError:
                continue
            if board_type not in datasets:
                datasets[board_type] = {}
            if heuristic not in datasets[board_type]:
                datasets[board_type][heuristic] = {}
            df = pd.read_csv(os.path.join(board_path, fname))
            datasets[board_type][heuristic][depth] = df
    return datasets


def plot_metric_single_heuristic(
    heuristic_dfs: Dict[int, pd.DataFrame],
    metric: str,
    heuristic_label: str,
    board_type: str,
    random_df: pd.DataFrame = None
) -> None:
    """
    Plot the percentage gain of the given metric for a single heuristic across depths for a specific board type,
    relative to the random baseline if random_df is provided.
    """
    all_depths = sorted(heuristic_dfs.keys())
    data = []
    stds = []
    random_mean = None
    random_std_val = None
    if random_df is not None:
        vals = random_df[metric]
        if metric == 'max_tile':
            vals = np.log2(vals)
        random_mean = vals.mean()
        random_std_val = vals.std()
    for d in all_depths:
        vals = heuristic_dfs[d][metric]
        if metric == 'max_tile':
            vals = np.log2(vals)
        mean = vals.mean()
        std = vals.std()
        if random_mean is not None and random_mean != 0:
            gain = ((mean - random_mean) / random_mean) * 100
            # std of percentage difference: sqrt((std1/random_mean)^2 + (std2*mean/random_mean^2)^2)
            gain_std = np.sqrt((std / random_mean) ** 2 + ((random_std_val * mean) / (random_mean ** 2)) ** 2) * 100
            data.append(gain)
            stds.append(gain_std)
        else:
            data.append(np.nan)
            stds.append(0)
    x = np.arange(len(all_depths))
    plt.figure(figsize=(max(10, len(all_depths) * 1.5), 6))
    plt.bar(x, data, width=0.6, yerr=stds, label=f'{heuristic_label} % gain', color='tab:green', capsize=5)
    plt.axhline(0, color='tab:red', linestyle='dashed', label='random baseline')
    plt.xticks(x, all_depths)
    plt.xlabel('Depth')
    title_label = 'log2(max_tile)' if metric == 'max_tile' else metric
    plt.ylabel(f'Percentage gain in {title_label} vs random (%)')
    plt.title(f'{board_type} - {heuristic_label}: Percentage gain in {title_label} over random across depths')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{board_type}_{heuristic_label}_percent_gain_vs_random_{title_label}_bar.png'))
    plt.close()


def main() -> None:
    datasets = load_datasets(DATA_DIR)
    metrics = ['max_tile', 'score']
    for board_type, heuristics in datasets.items():
        for heuristic, dfs in heuristics.items():
            if heuristic == 'random':
                continue
            random_df = heuristics['random'][0] if 'random' in heuristics and 0 in heuristics['random'] else None
            for metric in metrics:
                plot_metric_single_heuristic(
                    dfs,
                    metric=metric,
                    heuristic_label=heuristic,
                    board_type=board_type,
                    random_df=random_df
                )

if __name__ == '__main__':
    main() 