# 2048 AI with Deep Q-Learning

This project implements an AI agent that learns to play the game 2048 using Deep Q-Learning (DQN). The agent is trained to maximize its score by making optimal moves on a 4x4 grid.

## Project Structure

```
2048_AI/
│
├── game/
│   ├── __init__.py
│   ├── game_environment.py      # Gym environment for 2048
│   ├── game_logic.py            # Core game logic
│   └── game_renderer.py         # Pygame-based visualization
│
├── agent/
│   ├── __init__.py
│   ├── dqn_agent.py             # DQN agent implementation
│   └── replay_buffer.py         # Experience replay buffer
│
├── training/
│   ├── train.py                 # Training script
│   └── utils.py                 # Utility functions
│
├── graphics/
│   └── display.py               # Game visualization
│
├── results/
│   └── logs/                    # Training logs and models
│
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 2048_AI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the AI

To train the AI agent:

```bash
python -m training.train
```

The training script will:
- Train the agent for 10,000 episodes
- Save the best model and checkpoints every 500 episodes
- Display training progress every 1000 episodes
- Generate training plots and logs in the `results` directory

### Playing the Game Manually

To play the game manually:

```bash
python -m graphics.display
```

Use the arrow keys to move tiles:
- ↑: Move up
- →: Move right
- ↓: Move down
- ←: Move left

### Using a Trained Model

To use a trained model to play the game:

```python
from game.game_environment import Game2048Env
from agent.dqn_agent import DQNAgent

# Load the trained model
agent = DQNAgent(state_shape=(4, 4), action_size=4)
agent.load('results/models/best_model.h5')

# Create environment
env = Game2048Env()

# Play one episode
state = env.reset()
done = False
while not done:
    action = agent.act(state, env.game.get_valid_moves())
    state, reward, done, info = env.step(action)
```

## Training Parameters

The DQN agent is configured with the following parameters:
- Learning rate: 0.001
- Gamma (discount factor): 0.99
- Epsilon (exploration rate): 1.0 (decays over time)
- Memory size: 10,000
- Batch size: 32
- Target network update frequency: Every 10 episodes

## Results

The training results, including:
- Model checkpoints
- Training history
- Performance plots
- Best model

are saved in the `results` directory.

## Contributing

Feel free to submit issues and enhancement requests! 