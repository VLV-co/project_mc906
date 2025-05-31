import copy
from typing import Literal

import numpy as np
import pygame

from game import Game2048

class ExpectiMaxSearch:
    """Agent that plays 2048 using the Expectimax search algorithm."""

    def __init__(self, game_instance: Game2048, heuristic: Literal["empty_cells", "snake"], max_depth: int = 3):
        """
        Initializes the Expectimax agent.

        Args:
            game_instance (Game2048): Instance of the 2048 game.
            max_depth (int): Maximum search depth.
            heuristic (Literal): Heuristic to evaluate board states ('empty_cells' or 'snake').
        """
        self.game = game_instance
        self.max_depth = max_depth
        self.heuristic_type = heuristic
        self.snake_weights = self._generate_snake_weights() if heuristic == "snake" else None

    def get_best_move(self) -> int:
        """
        Computes the best move using the Expectimax algorithm.

        Returns:
            int: Number of a direction representing the best move.
        """
        best_score = -float("inf")
        best_move = None

        for direction in self.game.direction.values():
            temp_board = copy.deepcopy(self.game.board)
            _, moved = self.game._move_tiles(direction, temp_board)
            if not moved:
                continue
            value = self._expectimax(temp_board, depth=1, is_player=False)
            if value > best_score:
                best_score = value
                best_move = direction

        return best_move

    def _expectimax(self, board: np.ndarray, depth: int, is_player: bool) -> float:
        """
        Recursive Expectimax search function.

        Args:
            board (np.ndarray): The game board.
            depth (int): Current search depth.
            is_player (bool): Whether the current node is a player move.

        Returns:
            float: Expected utility of the board.
        """
        if depth == self.max_depth or not self._has_moves(board):
            return self._heuristic(board)

        if is_player:
            max_value = -float("inf")
            for direction in self.game.direction.values():
                new_board = copy.deepcopy(board)
                _, moved = self.game._move_tiles(direction, new_board)
                if not moved:
                    continue
                value = self._expectimax(new_board, depth + 1, is_player=False)
                max_value = max(max_value, value)
            return max_value
        else:
            empty = [(i, j) for i, j in self.game.board_coordinates if board[i][j] == 0]
            if not empty:
                return self._heuristic(board)

            total = 0
            for (i, j) in empty:
                for value, prob in [(2, 0.9), (4, 0.1)]:
                    new_board = copy.deepcopy(board)
                    new_board[i][j] = value
                    total += prob * self._expectimax(new_board, depth + 1, is_player=True)

            return total / len(empty)

    def _heuristic(self, board: np.ndarray) -> float:
        """
        Computes a heuristic evaluation of the board.   

        Args:
            board (np.ndarray): The game board.

        Returns:
            float: Heuristic score of the board.
        """
        flat = [board[i][j] for i, j in self.game.board_coordinates if board[i][j] != 'x']
        empty_cells = flat.count(0)
        max_tile = max(flat)

        if self.heuristic_type == "empty_cells":
            smoothness = -sum(
                abs(flat[i] - flat[i+1])
                for i in range(len(flat)-1)
                if flat[i] != 0 and flat[i+1] != 0
            )
            return empty_cells * 100 + max_tile * 10 + smoothness

        elif self.heuristic_type == "snake":
            score = 0
            for i, j in self.game.board_coordinates:
                if board[i][j] != 'x':
                    score += board[i][j] * self.snake_weights[i, j]
            return score

        else:
            raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")

    def _generate_snake_weights(self) -> np.ndarray:
        """
        Generates a weight matrix for the snake heuristic.

        Returns:
            np.ndarray: Snake weight matrix.
        """
        board = self.game.board
        heuristic_map = np.zeros_like(board, dtype=np.int64)
        num_cells = sum(1 for i, j in self.game.board_coordinates if board[i][j] != 'x') - 1

        for i in range(board.shape[0]):
            cols = range(board.shape[1]) if i % 2 == 0 else reversed(range(board.shape[1]))
            for j in cols:
                if board[i][j] != 'x':
                    heuristic_map[i, j] = 2 ** num_cells
                    num_cells -= 1

        return heuristic_map

    def _has_moves(self, board: np.ndarray) -> bool:
        """
        Checks if any valid moves are available.

        Args:
            board (np.ndarray): The game board.

        Returns:
            bool: True if there are possible moves, False otherwise.
        """
        for direction in self.game.direction.values():
            temp_board = copy.deepcopy(board)
            _, moved = self.game._move_tiles(direction, temp_board)
            if moved:
                return True
        return False


if __name__ == '__main__':
    # Run an example game with Expectimax agent
    game = Game2048(board_variant='triangle', size=4)
    agent = ExpectiMaxSearch(game_instance=game, max_depth=3, heuristic='snake')

    steps = 1

    while not game.game_over:
        # Let the ExpectiMax agent choose the best move
        direction = agent.get_best_move()
        if direction is None:
            print("No valid move found by AI.")
            break

        # Play the move in the real game
        _, score = game.play_step(direction)
        game.draw_board()

        print(f"Step {steps} | Move: {direction} | Score: {score}")

        steps += 1
        pygame.time.Clock().tick(5) 

    print("Game over!")
    print("Final score:", game.score)
    print("Max tile:", game._get_highest_block())

    pygame.quit()