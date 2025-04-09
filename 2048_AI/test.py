import torch
import numpy as np
from game import Game2048, Direction
from agent import Agent
import os

def test(n_games: int = 10) -> None:
    agent = Agent()
    game = Game2048()
    record = 0
    scores = []

    # Load trained model
    model_path = './model/model.pth'
    if not os.path.exists(model_path):
        print("No trained model found. Exiting test.")
        return

    agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.model.eval()
    print(f"Loaded model from: {model_path}")

    try:
        for i in range(1, n_games + 1):
            game.reset()
            state = agent.get_state(game)
            done = False

            while not done:
                action = agent.get_action(state, game=game, is_testing=True)
                final_move = agent.get_action(state)
                reward, done, score = game.play_step(final_move)
                state = agent.get_state(game)

            score = game.score
            scores.append(score)
            record = max(record, score)
            max_tile = int(np.max(game.board))

            print(f"Test Game {i} | Score: {score} | Max Tile: {max_tile}")

        avg_score = sum(scores) / len(scores)
        print(f"\n--- Test Summary ---")
        print(f"Games Played: {n_games}")
        print(f"Record Score: {record}")
        print(f"Average Score: {avg_score:.2f}")

    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")

if __name__ == '__main__':
    test(n_games=10)
