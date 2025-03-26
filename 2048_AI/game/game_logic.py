from typing import List, Tuple, Optional
import numpy as np
import random

class Game2048:
    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self) -> None:
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def move(self, direction: int) -> bool:
        """
        Move tiles in the specified direction
        direction: 0 (up), 1 (right), 2 (down), 3 (left)
        Returns: True if the move was valid and changed the board
        """
        original_board = self.board.copy()
        
        if direction in [0, 2]:  # Up or Down
            self.board = self.board.T
            
        if direction in [1, 3]:  # Right or Left
            self.board = np.fliplr(self.board)
            
        # Move and merge tiles
        for i in range(self.size):
            # Remove zeros
            row = self.board[i][self.board[i] != 0]
            # Merge adjacent equal numbers
            for j in range(len(row) - 1):
                if row[j] == row[j + 1]:
                    row[j] *= 2
                    self.score += row[j]
                    row[j + 1] = 0
            # Remove zeros again and pad with zeros
            row = row[row != 0]
            self.board[i] = np.pad(row, (0, self.size - len(row)), 'constant')
            
        if direction in [1, 3]:  # Right or Left
            self.board = np.fliplr(self.board)
            
        if direction in [0, 2]:  # Up or Down
            self.board = self.board.T
            
        return not np.array_equal(original_board, self.board)

    def is_game_over(self) -> bool:
        # Check for empty cells
        if 0 in self.board:
            return False
            
        # Check for possible merges
        for i in range(self.size):
            for j in range(self.size):
                current = self.board[i][j]
                # Check right
                if j < self.size - 1 and current == self.board[i][j + 1]:
                    return False
                # Check down
                if i < self.size - 1 and current == self.board[i + 1][j]:
                    return False
        return True

    def get_valid_moves(self) -> List[int]:
        valid_moves = []
        for direction in range(4):
            board_copy = self.board.copy()
            if self.move(direction):
                valid_moves.append(direction)
            self.board = board_copy
        return valid_moves

    def get_state(self) -> np.ndarray:
        return self.board.copy()

    def get_score(self) -> int:
        return self.score 