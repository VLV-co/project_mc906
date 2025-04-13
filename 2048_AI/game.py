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
        
        # Reward function weights
        self.alpha: float = 0.3  # Weight for score (decreased)
        self.beta: float = 0.2   # Weight for empty cells change (decreased)
        self.gamma: float = 0.5  # Weight for highest block change (increased)
        
        # Reward clipping range
        self.min_reward: float = -5.0  # Less penalization
        self.max_reward: float = 15.0  # More reward for good actions
        
        # Monitoring metrics
        self.moves_without_merge: int = 0
        self.max_tile_history: List[int] = []
        self.monotonicity_weight: float = 0.2  # Weight for monotonicity reward
        
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
        self.moves_without_merge = 0
        self.max_tile_history = []
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
    
    def is_valid_move(self, move: int) -> bool:
        """Check if a move is valid given a move index (0-3).
        
        Args:
            move: Integer representing the move (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        return self._is_valid_move(Direction(move + 1))
    
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
    
    def _count_empty_cells(self) -> int:
        return np.count_nonzero(self.board == 0)

    def _get_highest_block(self) -> int:
        return np.max(self.board)

    def _clip_reward(self, reward: float) -> float:
        return max(min(reward, self.max_reward), self.min_reward)

    def _calculate_monotonicity(self) -> float:
        """Calculate how monotonic the board is (increasing or decreasing values in rows/cols).
        Returns a value between -1 (worst) and 1 (best).
        """
        total_monotonicity = 0.0
        
        # Check rows
        for i in range(self.board_size):
            row = self.board[i]
            # Check left-to-right and right-to-left monotonicity
            left_to_right = sum(1 for j in range(self.board_size-1) if row[j] <= row[j+1] and row[j] != 0)
            right_to_left = sum(1 for j in range(self.board_size-1) if row[j] >= row[j+1] and row[j+1] != 0)
            row_monotonicity = max(left_to_right, right_to_left) / (self.board_size - 1)
            total_monotonicity += row_monotonicity
            
        # Check columns
        for j in range(self.board_size):
            col = self.board[:, j]
            # Check top-to-bottom and bottom-to-top monotonicity
            top_to_bottom = sum(1 for i in range(self.board_size-1) if col[i] <= col[i+1] and col[i] != 0)
            bottom_to_top = sum(1 for i in range(self.board_size-1) if col[i] >= col[i+1] and col[i+1] != 0)
            col_monotonicity = max(top_to_bottom, bottom_to_top) / (self.board_size - 1)
            total_monotonicity += col_monotonicity
            
        # Average and normalize to [-1, 1]
        return (total_monotonicity / (2 * self.board_size)) * 2 - 1

    def play_step(self, action: List[float]) -> Tuple[float, bool, int]:
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        direction = Direction(np.argmax(action) + 1)
        reward = 0.0
        
        if self._is_valid_move(direction):
            # Store state before move
            prev_empty_cells = self._count_empty_cells()
            prev_highest_block = self._get_highest_block()
            prev_score = self.score
            prev_monotonicity = self._calculate_monotonicity()
            
            # Make move
            score_increase, moved = self._move_tiles(direction, self.board)
            
            if moved:
                # Calculate state changes
                self.score += int(score_increase)
                curr_empty_cells = self._count_empty_cells()
                curr_highest_block = self._get_highest_block()
                curr_monotonicity = self._calculate_monotonicity()
                
                # Calculate components of reward
                score_reward = (self.score - prev_score) / 100.0  # Normalize score
                empty_cells_reward = (curr_empty_cells - prev_empty_cells) / self.board_size**2
                
                # Reward exponentially more for higher blocks
                if curr_highest_block > prev_highest_block:
                    highest_block_reward = (2 ** (np.log2(curr_highest_block) / 11.0)) - 1
                else:
                    highest_block_reward = 0
                
                # Reward for improved monotonicity
                monotonicity_reward = curr_monotonicity - prev_monotonicity
                
                # Calculate total reward using weights
                reward = (
                    self.alpha * score_reward +
                    self.beta * empty_cells_reward +
                    self.gamma * highest_block_reward +
                    self.monotonicity_weight * monotonicity_reward
                )
                
                # Bonus for merging higher tiles
                if curr_highest_block > prev_highest_block:
                    # Exponential reward based on the tile value reached
                    log_value = np.log2(curr_highest_block)
                    if log_value >= 8:  # 256 or higher
                        reward += log_value - 7  # Bonus increases with higher tiles
                    
                    self.moves_without_merge = 0
                    self.max_tile_history.append(curr_highest_block)
                else:
                    self.moves_without_merge += 1
                
                self._add_new_tile()
            else:
                reward = -0.5  # Reduced penalty for invalid moves
        else:
            reward = -1.0  # Reduced penalty for invalid moves
        
        # Check game over conditions
        self.game_over = len(self._get_valid_moves()) == 0
        
        if self.game_over:
            # Penalty based on the highest tile achieved
            max_tile = np.max(self.board)
            # Scale penalty - lower penalty for higher max tiles
            if max_tile >= 2048:
                reward = 20.0  # Big reward for reaching 2048
            elif max_tile >= 1024:
                reward = 10.0  # Reward for 1024
            elif max_tile >= 512:
                reward = 5.0   # Small reward for 512
            else:
                reward = -3.0  # Default penalty
        elif self.moves_without_merge > 30:  # Increased tolerance for moves without merge
            reward -= 0.5  # Reduced penalty
        
        # Clip reward to maintain stability
        reward = self._clip_reward(reward)
        
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