import pygame
import random
import numpy as np
from enum import Enum
from typing import Tuple, List, Optional

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Game2048:
    def __init__(self, width: int = 400, height: int = 500) -> None:
        self.width: int = width
        self.height: int = height
        self.board_size: int = 4
        self.cell_size: int = width // self.board_size
        self.margin: int = 10
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('2048 AI')
        self.font = pygame.font.Font(None, 36)
        
        self.colors: dict = {
            0: (204, 192, 179),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46)
        }
        
        self.reset()
    
    def reset(self) -> None:
        self.board: np.ndarray = np.zeros((self.board_size, self.board_size), dtype=int)
        self.score: int = 0
        self.game_over: bool = False
        self.frame_iteration: int = 0
        self._add_new_tile()
        self._add_new_tile()
    
    def _add_new_tile(self) -> None:
        empty_cells = [(i, j) for i in range(self.board_size) 
                      for j in range(self.board_size) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4
    
    def _get_valid_moves(self) -> List[Direction]:
        valid_moves = []
        for direction in Direction:
            if self._is_valid_move(direction):
                valid_moves.append(direction)
        return valid_moves
    
    def _is_valid_move(self, direction: Direction) -> bool:
        temp_board = self.board.copy()
        self._move_tiles(direction, temp_board)
        return not np.array_equal(temp_board, self.board)
    
    def _move_tiles(self, direction: Direction, board: np.ndarray) -> Tuple[int, bool]:
        score_increase = 0
        original_board = board.copy()
        
        # Rotate board according to direction
        if direction == Direction.UP:
            board = np.rot90(board, k=1)  # Rotate 90 degrees clockwise
        elif direction == Direction.DOWN:
            board = np.rot90(board, k=3)  # Rotate 270 degrees clockwise
        elif direction == Direction.RIGHT:
            board = np.rot90(board, k=2)  # Rotate 180 degrees
        
        # Process each row (now they're all effectively moving left)
        for i in range(board.shape[0]):
            # Remove zeros and get the non-zero numbers
            row = board[i][board[i] != 0]
            
            # Merge equal adjacent numbers
            merged_row = []
            skip_next = False
            for j in range(len(row)):
                if skip_next:
                    skip_next = False
                    continue
                if j < len(row) - 1 and row[j] == row[j + 1]:
                    merged_value = row[j] * 2
                    merged_row.append(merged_value)
                    score_increase += merged_value
                    skip_next = True
                else:
                    merged_row.append(row[j])
            
            # Pad with zeros to maintain board size
            merged_row.extend([0] * (board.shape[1] - len(merged_row)))
            board[i] = merged_row
        
        # Rotate back
        if direction == Direction.UP:
            board = np.rot90(board, k=3)
        elif direction == Direction.DOWN:
            board = np.rot90(board, k=1)
        elif direction == Direction.RIGHT:
            board = np.rot90(board, k=2)
        
        # Add bonus reward for non-empty tiles
        n_tiles = np.count_nonzero(board)
        score_increase += n_tiles * 0.1
        
        return score_increase, not np.array_equal(original_board, board)
    
    def play_step(self, action: List[float]) -> Tuple[int, bool, int]:
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        direction = Direction(np.argmax(action) + 1)
        reward = 0
        
        if self._is_valid_move(direction):
            score_increase, moved = self._move_tiles(direction, self.board)
            if moved:
                self.score += int(score_increase)  # Only add integer part to score
                reward = score_increase  # Keep floating point for reward
                self._add_new_tile()
        else:
            reward = -10
        
        self.game_over = len(self._get_valid_moves()) == 0
        
        if self.game_over:
            reward = -10
        elif np.max(self.board) >= 2048:
            reward = 100
            self.game_over = True
        
        self._update_ui()
        return reward, self.game_over, self.score
    
    def _update_ui(self) -> None:
        self.screen.fill((187, 173, 160))
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                value = self.board[i][j]
                color = self.colors.get(value, (237, 194, 46))
                
                rect = pygame.Rect(
                    j * self.cell_size + self.margin,
                    i * self.cell_size + self.margin,
                    self.cell_size - 2 * self.margin,
                    self.cell_size - 2 * self.margin
                )
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                if value != 0:
                    text_surface = self.font.render(str(value), True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)
        
        score_text = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
        self.screen.blit(score_text, (10, self.height - 40))
        
        pygame.display.flip()
    
    def get_state(self) -> np.ndarray:
        state = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                state.append(self.board[i][j])
        return np.array(state, dtype=float)

if __name__ == '__main__':
    game = Game2048()
    while True:
        action = [random.random() for _ in range(4)]
        reward, game_over, score = game.play_step(action)
        if game_over:
            print(f'Game Over! Score: {score}')
            break 