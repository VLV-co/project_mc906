import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal, Tuple, List
import numpy as np
import pygame
from game import Game2048


def mirror(board: np.ndarray) -> None:

    for i, row in enumerate(board):
        values = [v for v in row if v != 'x'][::-1]
        indices = [j for j, v in enumerate(row) if v != 'x']

        for j, val in zip(indices, values):
            row[j] = val


def merge(row: List) -> Tuple[List, int]:
    valid_values = [v for v in row if v != 'x']

    shifted = [v for v in valid_values if v != 0]
    score = 0  
    merged = []
    i = 0
    while i < len(shifted):
        if i + 1 < len(shifted) and shifted[i] == shifted[i + 1]:
            merged_value = shifted[i] * 2
            merged.append(merged_value)
            score += merged_value
            i += 2  # Skip the next element as it's been merged
        else:
            merged.append(shifted[i])
            i += 1
    
    return merged, score


def merge_left(board: np.ndarray) -> int:
    total_score = 0
    
    for i in range(len(board)):
        row, score = merge(board[i])
        total_score += score

        valid_positions = len([v for v in board[i] if v != 'x'])
        while len(row) < valid_positions:
            row.append(0)
        
        # Reconstruct the row with 'x' values in original positions
        result = []
        valid_idx = 0
        for v in board[i]:
            if v == 'x':
                result.append('x')
            else:
                result.append(row[valid_idx])
                valid_idx += 1
        board[i] = result
    
    return total_score


def move_tiles(direction: int, board: np.ndarray, board_variant: str, direction_map: dict) -> Tuple[int, bool]:

    original_board = board.copy()
    score = 0
    
    if direction == direction_map['left']:
        score = merge_left(board)
    elif direction == direction_map['right']:
        board[:] = np.fliplr(board)
        score = merge_left(board)
        board[:] = np.fliplr(board)
    elif board_variant == 'square':
        if direction == direction_map['up']:
            board[:] = np.rot90(board)
            score = merge_left(board)
            board[:] = np.rot90(board, k=3)
        elif direction == direction_map['down']:
            board[:] = np.rot90(board, k=3)
            score = merge_left(board)
            board[:] = np.rot90(board)
    else:  # triangular board
        if direction == direction_map['up_left']:
            mirror(board)
            board[:] = np.rot90(board)
            score = merge_left(board)
            board[:] = np.rot90(board, k=3)
            mirror(board)
        elif direction == direction_map['down_left']:
            mirror(board)
            board[:] = np.rot90(board, k=3)
            score = merge_left(board)
            board[:] = np.rot90(board)
            mirror(board)
        elif direction == direction_map['up_right']:
            board[:] = np.rot90(board)
            score = merge_left(board)
            board[:] = np.rot90(board, k=3)
        elif direction == direction_map['down_right']:
            board[:] = np.rot90(board, k=3)
            score = merge_left(board)
            board[:] = np.rot90(board)
    
    moved = not np.array_equal(original_board, board)
    return score, moved


def has_moves(board: np.ndarray, board_variant: str, direction_map: dict) -> bool:

    for direction in direction_map.values():
        temp_board = copy.deepcopy(board)
        _, moved = move_tiles(direction, temp_board, board_variant, direction_map)
        if moved:
            return True
    return False


def heuristic(board: np.ndarray, heuristic_type: str, snake_weights: np.ndarray, 
              board_coordinates: List) -> float:

    flat = [board[i][j] for i, j in board_coordinates if board[i][j] != 'x']
    empty_cells = flat.count(0)
    max_tile = max(flat) if flat else 0
    
    if heuristic_type == "empty_cells":
        smoothness = -sum(
            abs(flat[i] - flat[i+1])
            for i in range(len(flat)-1)
            if flat[i] != 0 and flat[i+1] != 0
        )
        return empty_cells * 100 + max_tile * 10 + smoothness
    
    elif heuristic_type == "snake":
        score = 0
        for i, j in board_coordinates:
            if board[i][j] != 'x':
                score += board[i][j] * snake_weights[i, j]
        return score
    
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")


def expectimax_worker(args: Tuple) -> float:

    (board, depth, is_player, max_depth, heuristic_type, 
     snake_weights, board_coordinates, board_variant, direction_map) = args
    
    if depth == max_depth or not has_moves(board, board_coordinates, board_variant, direction_map):
        return heuristic(board, heuristic_type, snake_weights, board_coordinates)
    
    if is_player:
        max_value = -float("inf")
        for direction in direction_map.values():
            new_board = copy.deepcopy(board)
            _, moved = move_tiles(direction, new_board, board_variant, direction_map)
            if not moved:
                continue
            
            args_new = (new_board, depth + 1, False, max_depth, heuristic_type,
                       snake_weights, board_coordinates, board_variant, direction_map)
            value = expectimax_worker(args_new)
            max_value = max(max_value, value)
        return max_value
    else:
        empty = [(i, j) for i, j in board_coordinates if board[i][j] == 0]
        if not empty:
            return heuristic(board, heuristic_type, snake_weights, board_coordinates)
        
        total = 0
        for (i, j) in empty:
            for value, prob in [(2, 0.9), (4, 0.1)]:
                new_board = copy.deepcopy(board)
                new_board[i][j] = value
                
                args_new = (new_board, depth + 1, True, max_depth, heuristic_type,
                           snake_weights, board_coordinates, board_variant, direction_map)
                total += prob * expectimax_worker(args_new)
        
        return total / len(empty)


def evaluate_move_worker(args: Tuple) -> Tuple[int, float]:

    (direction, board, max_depth, heuristic_type, snake_weights, 
     board_coordinates, board_variant, direction_map) = args
    
    temp_board = copy.deepcopy(board)
    _, moved = move_tiles(direction, temp_board, board_variant, direction_map)
    
    if not moved:
        return (direction, -float("inf"))
    
    worker_args = (temp_board, 1, False, max_depth, heuristic_type,
                   snake_weights, board_coordinates, board_variant, direction_map)
    value = expectimax_worker(worker_args)
    
    return (direction, value)


class ExpectiMaxSearch:

    def __init__(self, game_instance: Game2048, heuristic: Literal["empty_cells", "snake"], 
                 max_depth: int = 3, num_processes: int = None):

        self.game = game_instance
        self.max_depth = max_depth
        self.heuristic_type = heuristic
        self.num_processes = num_processes
        
        self.board_coordinates = list(game_instance.board_coordinates)
        self.board_variant = game_instance.board_variant
        self.direction_map = dict(game_instance.direction)
        
        self.snake_weights = self._generate_snake_weights() if heuristic == "snake" else None

    def get_best_move(self) -> int:

        current_board = copy.deepcopy(self.game.board)
        
        move_args = []
        for direction in self.direction_map.values():
            args = (direction, current_board, self.max_depth, self.heuristic_type,
                   self.snake_weights, self.board_coordinates, self.board_variant, self.direction_map)
            move_args.append(args)
        
        best_score = -float("inf")
        best_move = None
        
        # Parallel move evaluation
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            future_to_move = {executor.submit(evaluate_move_worker, args): args[0] 
                             for args in move_args}
            
            for future in as_completed(future_to_move):
                direction, value = future.result()
                if value > best_score:
                    best_score = value
                    best_move = direction
        
        return best_move

    def _generate_snake_weights(self) -> np.ndarray:

        board = self.game.board
        heuristic_map = np.zeros_like(board, dtype=np.int64)
        num_cells = sum(1 for i, j in self.board_coordinates if board[i][j] != 'x') - 1

        for i in range(board.shape[0]):
            cols = range(board.shape[1]) if i % 2 == 0 else reversed(range(board.shape[1]))
            for j in cols:
                if board[i][j] != 'x':
                    heuristic_map[i, j] = 2 ** num_cells
                    num_cells -= 1

        return heuristic_map


if __name__ == '__main__':
    game = Game2048(board_variant='hex', size=3)
    agent = ExpectiMaxSearch(game_instance=game, max_depth=5, heuristic='snake', num_processes=6)
    
    steps = 1
    while not game.game_over:
        direction = agent.get_best_move()
        if direction is None:
            print("No valid move found by AI.")
            break

        _, score = game.play_step(direction)
        game.draw_board()
        print(f"Step {steps} | Move: {direction} | Score: {score}")
        steps += 1

    print("Game over!")
    print("Final score:", game.score)
    print("Max tile:", game._get_highest_block())
    pygame.quit()