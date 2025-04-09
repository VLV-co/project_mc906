import torch
import numpy as np
from game import Game2048
from agent import Agent

def train() -> None:
    scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game2048()

    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # Treinamento curto (imediato)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                max_tile = int(np.max(game.board))
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games

                print(f'Game {agent.n_games} | Score {score} | Record {record} | Max tile {max_tile} | Epsilon {agent.epsilon:.3f}')

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == '__main__':
    train()
