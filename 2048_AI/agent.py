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
        self.epsilon_decay: float = 0.995  # Decay rate for exploration
        self.gamma: float = 0.9  # Discount rate
        self.memory: Deque = deque(maxlen=100_000)
        self.model = DQN(16, 256, 4)  # Input size: 4x4 board, Output size: 4 directions
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game: Game2048) -> np.ndarray:
        return game.get_state()

    def remember(self, state: np.ndarray, action: List[float], reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
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

    def get_action(self, state: np.ndarray) -> List[float]:
        # Random moves: tradeoff exploration / exploitation
        final_move = [0] * 4
        if random.random() < self.epsilon:  # Use probability instead of fixed threshold
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move 