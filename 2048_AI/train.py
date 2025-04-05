import torch
import numpy as np
from game import Game2048
from agent import Agent
from typing import List, Tuple
import os
from datetime import datetime
import json

def save_to_log(log_data: dict) -> None:
    """Save training results to a log file.
    
    Args:
        log_data: Dictionary containing training results and statistics
    """
    log_dir = './logs/training'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.json')
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"\nLog saved to: {log_file}")

def train() -> None:
    scores: List[int] = []
    total_score: int = 0
    record: int = 0
    agent = Agent()
    game = Game2048()
    
    # Training log data
    training_log = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'games': [],
        'final_stats': None
    }
    
    try:
        while True:
            # Initialize game log for this episode
            game_log = {
                'game_number': agent.n_games + 1,
                'moves_history': [],
                'epsilon': agent.epsilon,
                'final_board': None,
                'score': 0,
                'max_tile': 0,
                'total_moves': 0
            }
            
            # Get old state
            state_old = agent.get_state(game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            # Log the move
            moves_map = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
            move = final_move.index(1)
            move_info = {
                'move': moves_map[move],
                'reward': reward,
                'score': score,
                'board': game.board.tolist()  # Save board state after each move
            }
            game_log['moves_history'].append(move_info)
            
            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Train long memory (experience replay)
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                # Update game log
                game_log['final_board'] = game.board.tolist()
                game_log['score'] = score
                game_log['max_tile'] = int(np.max(game.board))
                game_log['total_moves'] = len(game_log['moves_history'])
                training_log['games'].append(game_log)

                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                
                print(f'Game {agent.n_games} | Score {score} | Record {record} | Mean Score {mean_score:.2f} | Epsilon {agent.epsilon:.3f}')

                # Save log every 20 games
                if agent.n_games % 20 == 0:
                    # Update final statistics
                    training_log['final_stats'] = {
                        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_games': agent.n_games,
                        'final_epsilon': agent.epsilon,
                        'record_score': record,
                        'average_score': float(total_score / agent.n_games),
                        'scores': scores
                    }
                    save_to_log(training_log)
                    # Start a new log for the next 20 games
                    training_log = {
                        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'games': [],
                        'final_stats': None
                    }

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save final log
        if training_log['games']:  # If there are any games in the current log
            training_log['final_stats'] = {
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_games': agent.n_games,
                'final_epsilon': agent.epsilon,
                'record_score': record,
                'average_score': float(total_score / agent.n_games),
                'scores': scores
            }
            save_to_log(training_log)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        if training_log['games']:
            save_to_log(training_log)

if __name__ == '__main__':
    train() 