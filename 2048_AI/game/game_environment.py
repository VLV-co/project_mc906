import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict, Any
from .game_logic import Game2048

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self) -> None:
        super(Game2048Env, self).__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        
        # Observation space is a 4x4 grid with values from 0 to 2048
        self.observation_space = spaces.Box(
            low=0,
            high=2048,
            shape=(4, 4),
            dtype=np.int32
        )
        
        self.game = Game2048()
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game and return the initial state."""
        self.game = Game2048()
        return self.game.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the game.
        
        Args:
            action: The action to take (0: up, 1: right, 2: down, 3: left)
            
        Returns:
            tuple: (state, reward, done, info)
        """
        # Store the previous score
        prev_score = self.game.get_score()
        
        # Make the move
        move_valid = self.game.move(action)
        
        # Calculate reward
        if not move_valid:
            reward = -10  # Penalty for invalid move
        else:
            # Reward based on score increase and tile values
            score_increase = self.game.get_score() - prev_score
            reward = score_increase + np.sum(self.game.get_state() > 0) * 0.1
            
        # Add new tile
        self.game.add_new_tile()
        
        # Check if game is over
        done = self.game.is_game_over()
        
        # Additional info
        info = {
            'score': self.game.get_score(),
            'valid_moves': self.game.get_valid_moves()
        }
        
        return self.game.get_state(), reward, done, info

    def render(self, mode: str = 'human') -> None:
        """Render the game state."""
        if mode == 'human':
            print("\n" + str(self.game.get_state()) + "\n")
        elif mode == 'rgb_array':
            # This would be implemented if we want to use the environment with computer vision
            raise NotImplementedError("RGB array rendering not implemented yet")

    def close(self) -> None:
        """Clean up resources."""
        pass 