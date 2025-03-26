import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json
from datetime import datetime

from ..game.game_environment import Game2048Env
from ..agent.dqn_agent import DQNAgent

class Trainer:
    def __init__(
        self,
        episodes: int = 10000,
        save_interval: int = 500,
        render_interval: int = 1000,
        model_path: str = "../results/models",
        log_path: str = "../results/logs"
    ) -> None:
        self.episodes = episodes
        self.save_interval = save_interval
        self.render_interval = render_interval
        self.model_path = model_path
        self.log_path = log_path
        
        # Create directories if they don't exist
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        # Initialize environment and agent
        self.env = Game2048Env()
        self.agent = DQNAgent(
            state_shape=(4, 4),
            action_size=4
        )
        
        # Training history
        self.history = {
            'scores': [],
            'avg_scores': [],
            'losses': [],
            'epsilons': []
        }

    def train(self) -> None:
        """Train the agent."""
        best_score = 0
        avg_score = 0
        
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Get valid moves
                valid_moves = self.env.game.get_valid_moves()
                
                # Choose action
                action = self.agent.act(state, valid_moves)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Train agent
                metrics = self.agent.train(state, action, reward, next_state, done, valid_moves)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Update target model if needed
                if episode % self.agent.target_update == 0:
                    self.agent.update_target_model()
            
            # Update history
            self.history['scores'].append(info['score'])
            self.history['losses'].append(metrics['loss'])
            self.history['epsilons'].append(metrics['epsilon'])
            
            # Calculate average score
            if episode >= 100:
                avg_score = np.mean(self.history['scores'][-100:])
                self.history['avg_scores'].append(avg_score)
            
            # Save best model
            if info['score'] > best_score:
                best_score = info['score']
                self.agent.save(os.path.join(self.model_path, 'best_model.h5'))
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                self.agent.save(os.path.join(self.model_path, f'model_episode_{episode}.h5'))
                self._save_history()
            
            # Render game
            if episode % self.render_interval == 0:
                print(f"Episode: {episode}")
                print(f"Score: {info['score']}")
                print(f"Average Score: {avg_score:.2f}")
                print(f"Epsilon: {metrics['epsilon']:.2f}")
                print(f"Loss: {metrics['loss']:.4f}")
                print("-" * 50)
        
        # Save final model and history
        self.agent.save(os.path.join(self.model_path, 'final_model.h5'))
        self._save_history()
        
        # Plot training results
        self._plot_results()

    def _save_history(self) -> None:
        """Save training history to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(self.log_path, f'training_history_{timestamp}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {
            'scores': [float(x) for x in self.history['scores']],
            'avg_scores': [float(x) for x in self.history['avg_scores']],
            'losses': [float(x) for x in self.history['losses']],
            'epsilons': [float(x) for x in self.history['epsilons']]
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_dict, f)

    def _plot_results(self) -> None:
        """Plot training results."""
        plt.figure(figsize=(15, 10))
        
        # Plot scores
        plt.subplot(2, 2, 1)
        plt.plot(self.history['scores'])
        plt.title('Scores per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Plot average scores
        plt.subplot(2, 2, 2)
        plt.plot(self.history['avg_scores'])
        plt.title('Average Scores (Last 100 Episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        
        # Plot losses
        plt.subplot(2, 2, 3)
        plt.plot(self.history['losses'])
        plt.title('Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        # Plot epsilon
        plt.subplot(2, 2, 4)
        plt.plot(self.history['epsilons'])
        plt.title('Epsilon per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.log_path, f'training_results_{timestamp}.png'))
        plt.close()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train() 