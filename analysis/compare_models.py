import os
import re
from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Regex to extract heuristic and depth from filename
def parse_filename(filename: str) -> Tuple[str, int]:
    match = re.match(r"expectimax_([a-z_]+)_depth_(\d+)_square_size_\d+\.csv", filename)
    if match:
        heuristic, depth = match.groups()
        return heuristic, int(depth)
    elif filename.startswith('random'):
        return 'random', 0
    raise ValueError(f"Unrecognized filename: {filename}")

def load_datasets(data_dir: str) -> Dict[str, Dict[int, pd.DataFrame]]:
    datasets: Dict[str, Dict[int, pd.DataFrame]] = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        heuristic, depth = parse_filename(fname)
        if heuristic not in datasets:
            datasets[heuristic] = {}
        df = pd.read_csv(os.path.join(data_dir, fname))
        datasets[heuristic][depth] = df
    return datasets


def plot_metric_all_models(datasets: Dict[str, Dict[int, pd.DataFrame]], metric: str) -> None:
    # Prepare data: rows=depths, columns=heuristics
    all_heuristics = sorted([h for h in datasets if h != 'random'])
    all_depths = sorted(set(d for h in datasets for d in datasets[h] if h != 'random'))
    data = {h: [] for h in all_heuristics}
    stds = {h: [] for h in all_heuristics}
    for d in all_depths:
        for h in all_heuristics:
            if d in datasets[h]:
                vals = datasets[h][d][metric]
                if metric == 'max_tile':
                    vals = np.log2(vals)
                data[h].append(vals.mean())
                stds[h].append(vals.std())
            else:
                data[h].append(np.nan)
                stds[h].append(0)
    x = np.arange(len(all_depths))
    width = 0.8 / len(all_heuristics) if all_heuristics else 0.8
    plt.figure(figsize=(max(10, len(all_depths) * 1.5), 6))
    palette = sns.color_palette('viridis', len(all_heuristics))
    for i, h in enumerate(all_heuristics):
        plt.bar(x + i * width, data[h], width=width, yerr=stds[h], label=h, color=palette[i], capsize=5)
    plt.xticks(x + width * (len(all_heuristics) - 1) / 2, all_depths)
    plt.xlabel('Depth')
    title_label = 'log2(max_tile)' if metric == 'max_tile' else metric
    plt.ylabel(title_label)
    plt.title(f'Comparison of {title_label} across depths')
    plt.legend(title='Heuristic')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{title_label}_bar_all_models.png'))
    plt.close()

def main() -> None:
    datasets = load_datasets(DATA_DIR)
    plot_metric_all_models(datasets, 'max_tile')
    plot_metric_all_models(datasets, 'score')

if __name__ == '__main__':
    main() 