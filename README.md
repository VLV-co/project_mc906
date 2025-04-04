# 2048 AI with Deep Q-Learning

This project implements an AI agent that learns to play the 2048 game using Deep Q-Learning (DQN). The agent learns to make optimal moves in the 4x4 grid to achieve the highest possible score.

## Project Structure

```
2048-AI/
├── game.py           # Game environment implementation
├── model.py          # DQN neural network model
├── agent.py          # DQN agent implementation
├── train.py          # Training script
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

## Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

## How It Works

1. **Game Environment (game.py)**:
   - Implements the 2048 game mechanics
   - Uses Pygame for visualization
   - Provides an interface for the AI agent to interact with

2. **DQN Model (model.py)**:
   - Neural network architecture for Q-learning
   - Input: 4x4 game board state (16 values)
   - Hidden layers: 256 neurons each
   - Output: Q-values for 4 possible actions (up, down, left, right)

3. **Agent (agent.py)**:
   - Implements the DQN agent with experience replay
   - Uses epsilon-greedy strategy for exploration
   - Manages the training process and decision making

4. **Training (train.py)**:
   - Main training loop
   - Plots training progress
   - Saves best performing models

## Training Process

The agent uses the following techniques:
- Experience replay buffer (100,000 experiences)
- Epsilon-greedy exploration strategy
- Reward shaping based on score and valid moves
- Automatic model saving for best performances

## Running the Project

1. Train the agent:
```bash
python train.py
```

2. The training script will:
   - Display the game board
   - Show real-time training statistics
   - Plot the scores and mean scores
   - Save the best performing model

## Model Architecture

- Input layer: 16 neurons (4x4 board)
- Hidden layer 1: 256 neurons with ReLU
- Hidden layer 2: 256 neurons with ReLU
- Output layer: 4 neurons (one for each direction)

## Rewards Structure

- Score increase: Positive reward equal to the merge value
- Invalid move: -10 reward
- Game over: -10 reward
- Reaching 2048: +100 reward

## Performance

The agent typically learns to:
1. Avoid invalid moves
2. Keep high-value tiles in corners
3. Merge similar tiles effectively
4. Build up to higher numbers systematically 