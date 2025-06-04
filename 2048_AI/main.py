import argparse
import os
import random
import time
from typing import Dict, Optional

import pandas as pd

from expectimax_search import ExpectiMaxSearch
from game import Game2048

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for running 2048 experiments.

    Returns:
        argparse.Namespace: Parsed arguments including agent type, heuristic, depth, number of games, 
        board variant, board size, and output directory.
    """
    parser = argparse.ArgumentParser(description="Run 2048 Experiment")

    parser.add_argument("-agent", type=str, default="expectimax", choices=["expectimax", "random"],
                        help="Type of agent to use: 'expectimax' or 'random'")
    parser.add_argument("-heuristic", type=str, default="snake",
                        help="Heuristic to use for expectimax: empty_cells, or snake")
    parser.add_argument("-depth", type=int, default=3,
                        help="Max depth for expectimax search")
    parser.add_argument("-n_games", type=int, default=100,
                        help="Number of games to run")
    parser.add_argument("-board_variant", type=str, default="square", choices=["triangle", "square", "hex"],
                        help="Game board variant: 'triangle', 'square' or 'hex')")
    parser.add_argument("-size", type=int, default=4,
                        help="Board size (NxN)")
    parser.add_argument("-output_dir", type=str, default="results",
                        help="Directory to save experiment logs and results")
    return parser.parse_args()

def play_game(
        game: Game2048,
        game_number: int,
        log_file_handle,
        agent_type: str,
        heuristic: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> Dict[str, int]:

    """
    Runs a single game of 2048 using the specified agent and logs key milestones.

    Args:
        game (Game2048): The 2048 game instance.
        game_number (int): Identifier for the current game (used in logs).
        log_file_handle (file object): Open file handle for logging.
        agent_type (str): Type of agent to use ("expectimax" or "random").
        heuristic (Optional[str]): Heuristic strategy used by the expectimax agent. Defaults to None.
        depth (Optional[int]): Maximum depth for the expectimax search. Defaults to None.

    Returns:
        Dict[str, int]: Dictionary containing game number, total move count, final score
        maximum tile reached, and average time per move (in seconds).
    """
    move_count = 0
    previous_max_tile = 0
    move_durations = []

    agent = None
    if agent_type == "expectimax":
        agent = ExpectiMaxSearch(game_instance=game, max_depth=depth, heuristic=heuristic, num_processes=6)

    while not game.game_over:
        start_time = time.time()

        if agent_type == "random":
            valid_directions = [d for d in game.direction.values() if game._is_valid_move(d)]
            if not valid_directions:
                break
            direction = random.choice(valid_directions)
        else:
            direction = agent.get_best_move()
            if direction is None:
                break
        
        duration = time.time() - start_time
        move_durations.append(duration)

        # Play the move in the real game
        _, score = game.play_step(direction)
        game.draw_board()

        move_count += 1
        current_max_tile = game.get_max_tile()

        if current_max_tile > previous_max_tile:
            log_file_handle.write(
                f"Game number: {game_number:<4} | "
                f"Move count: {move_count:<5} | "
                f"Current score: {score:<6} | "
                f"Current max tile: {current_max_tile:<5} | "
                f"Move time: {duration:>7.4f}s\n"
            )
            previous_max_tile = current_max_tile

    avg_move_time = sum(move_durations) / len(move_durations) if move_durations else 0

    return {
        "game_number": game_number,
        "move_count": move_count,
        "score": game.score,
        "max_tile": game.get_max_tile(),
        "avg_move_time_sec": avg_move_time
    }


def main():
    args = parse_args()

    if args.agent == 'expectimax':
        exp_name = f"{args.agent}_{args.heuristic}_depth_{args.depth}_{args.board_variant}_size_{args.size}"
    elif args.agent == 'random':
        exp_name = f"{args.agent}_{args.board_variant}_size_{args.size}"
    else:
        raise ValueError(f"Unknown agent type: '{args.agent}'. Use 'expectimax' or 'random'.")

    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)

    experiment_log_path = os.path.join(log_dir, "experiment.log")
    with open(experiment_log_path, "w") as log_file:
        if args.agent == 'expectimax':
            print(
                f"Running {args.n_games} games with: "
                f"agent={args.agent}, heuristic={args.heuristic}, depth={args.depth}"
            )
        else:
            print(f"Running {args.n_games} games with: agent={args.agent}")

        print(f"Logging to: {log_dir}")

        game = Game2048(board_variant=args.board_variant, size=args.size)
        game_data = []

        for i in range(1, args.n_games + 1):
            game.reset()
            result = play_game(
                game,
                game_number=i,
                log_file_handle=log_file,
                agent_type=args.agent,
                heuristic=args.heuristic if args.agent == 'expectimax' else None,
                depth=args.depth if args.agent == 'expectimax' else None,
            )
            game_data.append(result)

            print(
                f"Finished game {i:<3} | "
                f"Moves: {result['move_count']:<5} | "
                f"Score: {result['score']:<6} | "
                f"Max tile: {result['max_tile']:<5} | "
                f"Average move time: {result['avg_move_time_sec']:>7.4f}s"
            )

            pd.DataFrame(game_data).to_csv(os.path.join(log_dir, "summary.csv"), index=False)

if __name__ == "__main__":
    main()
