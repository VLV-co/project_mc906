import torch
import pygame
import time
from game import Game2048
from model import DQN
from agent import Agent
from typing import List, Tuple
import numpy as np
import os
from datetime import datetime
import json

def save_to_log(log_data: dict) -> None:
    """Save test results to a log file.
    
    Args:
        log_data: Dictionary containing test results and statistics
    """
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'test_log_{timestamp}.json')
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"\nLog saved to: {log_file}")

def test_model(model_path: str = './model/model.pth', n_games: int = 10, delay: float = 0.1) -> None:
    """
    Test a trained model on the 2048 game
    
    Args:
        model_path: Path to the saved model
        n_games: Number of games to play
        delay: Delay between moves in seconds
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize game and model
    try:
        game = Game2048()
        model = DQN(16, 256, 4)  # Same architecture as training
        
        # Load model weights safely
        if torch.cuda.is_available():
            state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(state_dict)
            model = model.cuda()
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        print(f"Successfully loaded model from {model_path}")
        print(f"Testing model on {n_games} games...")
        print("-------------------")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Statistics and logging
    scores: List[int] = []
    max_tiles: List[int] = []
    moves: List[int] = []
    games_log: List[dict] = []
    
    try:
        # Play n_games
        for i in range(n_games):
            game.reset()
            game_moves = 0
            game_log = {
                'game_number': i + 1,
                'moves_history': [],
                'final_board': None,
                'score': 0,
                'max_tile': 0,
                'total_moves': 0
            }
            
            while not game.game_over:
                # Get state
                state = game.get_state()
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                if torch.cuda.is_available():
                    state_tensor = state_tensor.cuda()
                
                # Get move from model
                with torch.no_grad():
                    prediction = model(state_tensor)
                    move = torch.argmax(prediction).item()
                
                # Convert to one-hot
                final_move = [0] * 4
                final_move[move] = 1
                
                # Make move and get result
                reward, done, score = game.play_step(final_move)
                
                # Log the move
                moves_map = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
                move_info = {
                    'move': moves_map[move],
                    'reward': reward,
                    'score': score
                }
                game_log['moves_history'].append(move_info)
                
                if reward == -10:  # Invalid move
                    print('Warning: Invalid move detected')
                
                if done:
                    print('Game Over! No more valid moves available.')
                    print(f'Final board state:\n{game.board}')
                    # Save final board state
                    game_log['final_board'] = game.board.tolist()
                
                game_moves += 1
                
                # Update display
                pygame.display.update()
                
                # Add delay to visualize
                time.sleep(delay)
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            
            # Record statistics
            scores.append(score)
            max_tile = np.max(game.board)
            max_tiles.append(max_tile)
            moves.append(game_moves)
            
            # Update game log
            game_log['score'] = score
            game_log['max_tile'] = int(max_tile)
            game_log['total_moves'] = game_moves
            games_log.append(game_log)
            
            print(f'Game {i+1}/{n_games}:')
            print(f'Score: {score}')
            print(f'Max tile: {max_tile}')
            print(f'Moves: {game_moves}')
            print('-------------------')
        
        # Prepare final statistics
        final_stats = {
            'model_path': model_path,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'number_of_games': n_games,
            'average_score': float(sum(scores)/len(scores)),
            'max_score': max(scores),
            'average_max_tile': float(sum(max_tiles)/len(max_tiles)),
            'max_tile_achieved': max(max_tiles),
            'average_moves': float(sum(moves)/len(moves)),
            'games': games_log
        }
        
        # Save log
        save_to_log(final_stats)
        
        # Print final statistics
        print('\nFinal Statistics:')
        print(f'Average Score: {final_stats["average_score"]:.2f}')
        print(f'Max Score: {final_stats["max_score"]}')
        print(f'Average Max Tile: {final_stats["average_max_tile"]:.2f}')
        print(f'Max Tile Achieved: {final_stats["max_tile_achieved"]}')
        print(f'Average Moves per Game: {final_stats["average_moves"]:.2f}')
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
    finally:
        pygame.quit()

if __name__ == '__main__':
    test_model() 