import torch
import random
import numpy as np
from collections import deque
from game import Game2048, Direction
from model import DQN, QTrainer
from typing import List, Tuple, Deque

class Agent:
    def __init__(self) -> None:
        self.n_games: int = 0
        self.epsilon: float = 1.0  # Randomness - start with 100% exploration
        self.epsilon_min: float = 0.01  # Minimum exploration rate
        self.epsilon_decay: float = 0.999  # Decay rate for exploration
        self.gamma: float = 0.9  # Discount rate
        self.memory: Deque = deque(maxlen=100_000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(16, 256, 4)  # Input size: 4x4 board, Output size: 4 directions
        self.model = self.model.to(self.device)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game: Game2048) -> np.ndarray:
        return game.get_state()

    def remember(self, state: np.ndarray, action: List[float], reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > 100:
            mini_sample = random.sample(self.memory, 100)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        # Decay epsilon after each episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_short_memory(self, state: np.ndarray, action: List[float], reward: float, 
                         next_state: np.ndarray, done: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray, game: Game2048 = None, is_testing: bool = False) -> List[float]:
        # Random moves: tradeoff exploration / exploitation
        final_move = [0] * 4
        
        if is_testing and game is not None:
            # During testing, only consider valid moves
            valid_moves = []
            for move in range(4):
                if game.is_valid_move(move):
                    valid_moves.append(move)
            
            if not valid_moves:
                # If no valid moves, return any move (game will end anyway)
                move = random.randint(0, 3)
            else:
                state0 = torch.tensor(state, dtype=torch.float).to(self.device)
                prediction = self.model(state0)
                prediction = prediction.detach().cpu().numpy()  # Move back to CPU for numpy operations
                
                # Filter predictions to only consider valid moves
                valid_predictions = [(move, prediction[move]) for move in valid_moves]
                move = max(valid_predictions, key=lambda x: x[1])[0]
            
            final_move[move] = 1
            return final_move
            
        # During training, use epsilon-greedy strategy
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)
            prediction = prediction.detach().cpu().numpy()  # Move back to CPU for numpy operations
            move = np.argmax(prediction)
            final_move[move] = 1

        return final_move 