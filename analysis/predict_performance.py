from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from typing import Tuple


def get_alpha_beta(df: pd.DataFrame) -> Tuple[float, float]:
    X = df[['possible_moves', 'connection_degree']]
    y = df['phi'] - df['cells_number']
    model = LinearRegression().fit(X, y)
    alpha, beta = model.coef_
    return alpha, beta


def get_empirical_stats(csv_path: str) -> Tuple[float, float]:
    df = pd.read_csv(csv_path)
    log2_max_tile = np.log2(df['max_tile'])
    return float(np.mean(log2_max_tile)), float(np.std(log2_max_tile))


def main() -> None:
    # Constants
    P2 = 0.9
    P4 = 0.1
    mu = 0.05
    lambda_ = 0.0
    H = 0.0
    # Tile generation
    E_T_new = 2 * P2 + 4 * P4
    P = 1 / E_T_new
    print(f"E[T_new]: {E_T_new:.3f}, P: {P:.3f}")

    # Load topology data
    topo_df = pd.read_csv('data/aggregated/topologies_phi.csv')
    # Map: 0=hex, 1=square, 2=triangle (from aggregate_topology_phi.py)
    topo_names = ['hex', 'square', 'triangle']
    topo_params = {}
    for i, name in enumerate(topo_names):
        row = topo_df.iloc[i]
        topo_params[name] = {
            'N': row['cells_number'],
            'D': row['possible_moves'],
            'dbar': row['connection_degree'],
            'phi': row['phi']
        }

    # Fit alpha, beta
    alpha, beta = get_alpha_beta(topo_df)
    print(f"Fitted alpha: {alpha:.4f}, beta: {beta:.4f}")

    def f(N: float, D: float, dbar: float) -> float:
        return N + alpha * D + beta * dbar

    # Empirical stats for best snake model (depth 4) for each topology
    best_snake_csvs = {
        'square': 'data/square/expectimax_snake_depth_4_square_size_4.csv',
        'hex': 'data/hex/expectimax_snake_depth_4_hex_size_3.csv',
        'triangle': 'data/triangle/expectimax_snake_depth_4_triangle_size_4.csv',
    }
    stats = {}
    for name in topo_names:
        mean_log2, std_log2 = get_empirical_stats(best_snake_csvs[name])
        stats[name] = {'mean_log2': mean_log2, 'std_log2': std_log2}
        print(f"{name.title()} mean log2(max_tile): {mean_log2:.4f}, std: {std_log2:.4f}")

    # Compute Phi for each
    for name in topo_names:
        mean_log2 = stats[name]['mean_log2']
        std_log2 = stats[name]['std_log2']
        phi = mean_log2 * (1 + mu * std_log2)
        stats[name]['phi'] = phi
        print(f"{name.title()} Phi: {phi:.4f}")

    # Compute D_topo for each
    for name in topo_names:
        N = topo_params[name]['N']
        D = topo_params[name]['D']
        dbar = topo_params[name]['dbar']
        phi = stats[name]['phi']
        D_topo = f(N, D, dbar) / phi * P * (1 + lambda_ * H)
        stats[name]['D_topo'] = D_topo
        print(f"{name.title()} D_topo: {D_topo:.4f}")

    # Use square as baseline
    D_ref = stats['square']['D_topo']
    print(f"\nUsing Square as baseline (D_ref = {D_ref:.4f})")

    for name in ['hex', 'triangle']:
        N = topo_params[name]['N']
        D = topo_params[name]['D']
        dbar = topo_params[name]['dbar']
        std_log2 = stats[name]['std_log2']
        # Predict Phi for new topology
        Phi_pred = f(N, D, dbar) / D_ref * P * (1 + lambda_ * H + mu * std_log2)
        # Predict expected log2(max_tile)
        E_log2_pred = Phi_pred / (1 + mu * std_log2)
        print(f"\n{name.title()} prediction:")
        print(f"  Predicted Phi: {Phi_pred:.4f}")
        print(f"  Predicted E[log2(max_tile)]: {E_log2_pred:.4f}")

    # --- Extension: Predict for varying N (cells_number) for hex and triangle ---
    print("\nPredictions for varying N (cells_number) with fixed D and dbar:")
    for name in ['hex', 'triangle']:
        N_orig = topo_params[name]['N']
        D = topo_params[name]['D']
        dbar = topo_params[name]['dbar']
        std_log2 = stats[name]['std_log2']
        print(f"\n{name.title()} (D={D}, dbar={dbar}, std_log2={std_log2:.4f}):")
        for N_var in range(int(N_orig) - 4, int(N_orig) + 5):
            if N_var <= 0:
                continue
            Phi_pred = f(N_var, D, dbar) / D_ref * P * (1 + lambda_ * H + mu * std_log2)
            E_log2_pred = Phi_pred / (1 + mu * std_log2)
            print(f"  N={N_var:2d}: Predicted E[log2(max_tile)] = {E_log2_pred:.4f}")

if __name__ == '__main__':
    main()
