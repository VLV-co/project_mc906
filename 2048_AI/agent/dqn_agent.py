from typing import List, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import deque
import os

class DQNAgent:
    def __init__(
        self,
        state_shape: Tuple[int, int],
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 10
    ) -> None:
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self) -> tf.keras.Model:
        """Build the neural network model."""
        model = models.Sequential([
            layers.Input(shape=self.state_shape),
            layers.Conv2D(64, (2, 2), activation='relu'),
            layers.Conv2D(128, (2, 2), activation='relu'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def update_target_model(self) -> None:
        """Update the target model with the current model's weights."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, valid_moves: List[int]) -> int:
        """Choose an action using epsilon-greedy policy."""
        if random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        # Reshape state for prediction
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        
        # Filter out invalid moves
        act_values[0, [i for i in range(self.action_size) if i not in valid_moves]] = float('-inf')
        
        return np.argmax(act_values[0])

    def replay(self) -> float:
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Reshape states for prediction
        states = np.expand_dims(states, axis=1)
        next_states = np.expand_dims(next_states, axis=1)

        # Get current Q values
        current_q_values = self.model.predict(states, verbose=0)
        
        # Get next Q values from target model
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        history = self.model.fit(states, current_q_values, epochs=1, verbose=0)
        return history.history['loss'][0]

    def train(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool, valid_moves: List[int]) -> Dict[str, float]:
        """Train the agent on a single step."""
        # Store experience
        self.remember(state, action, reward, next_state, done)
        
        # Train on batch
        loss = self.replay()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return {'loss': loss, 'epsilon': self.epsilon}

    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        self.model.save(filepath)

    def load(self, filepath: str) -> None:
        """Load the model from a file."""
        if os.path.exists(filepath):
            self.model = models.load_model(filepath)
            self.target_model = models.load_model(filepath) 