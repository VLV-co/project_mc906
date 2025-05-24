import numpy as np
from game import Game2048
from typing import List, Tuple
import concurrent.futures

NUM_DIRECTIONS = 6  # Novo jogo tem 6 direções
GRID_SIZE = 5

# --- Constantes e helpers do game.py ---
GRID_COORDENATES = [(i, j) for i, cols in enumerate([range(0, 3), range(0, 4), range(0, 5), range(0, 4), range(0, 3)]) for j in cols]
MAIN_DIAGONAL = [
    [(0, 2), (1, 3), (2, 4)],
    [(0, 1), (1, 2), (2, 3), (3, 3)],
    [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2)],
    [(1, 0), (2, 1), (3, 1), (4, 1)],
    [(2, 0), (3, 0), (4, 0)]
]
SECONDARY_DIAGONAL = [
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 0)],
    [(0, 2), (1, 2), (2, 2), (3, 1), (4, 0)],
    [(1, 3), (2, 3), (3, 2), (4, 1)],
    [(2, 4), (3, 3), (4, 2)]
]


def _merge(line: list[int]) -> tuple[list[int], int]:
    non_zeros = [x for x in line if x != 0]
    line = non_zeros + [0] * (len(line) - len(non_zeros))
    score = 0
    for i in range(len(line) - 1):
        if line[i] != 0 and line[i] == line[i + 1]:
            line[i] *= 2
            score += line[i]
            line[i + 1] = 0
    non_zeros = [x for x in line if x != 0]
    line = non_zeros + [0] * (len(line) - len(non_zeros))
    return line, score


def move_board(board: list[list[int]], direction: int) -> tuple[list[list[int]], int, bool]:
    """
    Realiza o movimento na matriz board na direção especificada, retornando a nova matriz, score ganho e se houve movimento.
    """
    import copy
    new_board = copy.deepcopy(board)
    moved = False
    score = 0
    if direction == 0:  # main diagonal up
        for k in range(len(new_board)):
            coord = MAIN_DIAGONAL[k]
            transversal = [new_board[i][j] for (i, j) in coord]
            new_transversal, add_score = _merge(transversal)
            for l in range(len(transversal)):
                if transversal[l] != new_transversal[l]:
                    moved = True
                    i, j = coord[l]
                    new_board[i][j] = new_transversal[l]
            score += add_score
    elif direction == 1:  # secondary diagonal up
        for k in range(len(new_board)):
            coord = SECONDARY_DIAGONAL[k]
            transversal = [new_board[i][j] for (i, j) in coord]
            new_transversal, add_score = _merge(transversal)
            for l in range(len(transversal)):
                if transversal[l] != new_transversal[l]:
                    moved = True
                    i, j = coord[l]
                    new_board[i][j] = new_transversal[l]
            score += add_score
    elif direction == 2:  # right
        for k in range(len(new_board)):
            row = [new_board[i][j] for (i, j) in GRID_COORDENATES if i == k]
            row.reverse()
            new_row, add_score = _merge(row)
            new_row.reverse()
            for j in range(len(row)):
                if new_board[k][j] != new_row[j]:
                    moved = True
                    new_board[k][j] = new_row[j]
            score += add_score
    elif direction == 3:  # main diagonal down
        for k in range(len(new_board)):
            coord = MAIN_DIAGONAL[k]
            transversal = [new_board[i][j] for (i, j) in coord]
            transversal.reverse()
            new_transversal, add_score = _merge(transversal)
            new_transversal.reverse()
            for l in range(len(transversal)):
                i, j = coord[l]
                if new_board[i][j] != new_transversal[l]:
                    moved = True
                    new_board[i][j] = new_transversal[l]
            score += add_score
    elif direction == 4:  # secondary diagonal down
        for k in range(len(new_board)):
            coord = SECONDARY_DIAGONAL[k]
            transversal = [new_board[i][j] for (i, j) in coord]
            transversal.reverse()
            new_transversal, add_score = _merge(transversal)
            new_transversal.reverse()
            for l in range(len(transversal)):
                i, j = coord[l]
                if new_board[i][j] != new_transversal[l]:
                    moved = True
                    new_board[i][j] = new_transversal[l]
            score += add_score
    elif direction == 5:  # left
        for k in range(len(new_board)):
            row = [new_board[i][j] for (i, j) in GRID_COORDENATES if i == k]
            new_row, add_score = _merge(row)
            for j in range(len(row)):
                if new_board[k][j] != new_row[j]:
                    moved = True
                    new_board[k][j] = new_row[j]
            score += add_score
    return new_board, score, moved

def is_valid_move_board(board: list[list[int]], direction: int) -> bool:
    _, _, moved = move_board(board, direction)
    return moved

def score_move_for_expectimax(args: tuple[list[list[int]], int, int]) -> tuple[int, float]:
    board, move, max_depth = args
    new_board, _, moved = move_board(board, move)
    if moved:
        agent = ExpectimaxAgent(max_depth)
        score = agent.expectimax(new_board, max_depth - 1, False)
        return (move, score)
    else:
        return (move, float('-inf'))

class ExpectimaxAgent:
    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth: int = max_depth

    def get_state(self, game: Game2048) -> List[List[int]]:
        return [row[:] for row in game.grid]

    def get_action(self, state: List[List[int]], game: Game2048 = None) -> List[float]:
        assert game is not None, 'ExpectimaxAgent precisa de uma instância de Game2048.'
        board = [row[:] for row in game.grid]
        best_move = self.expectimax_decision(board)
        final_move = [0] * NUM_DIRECTIONS
        if best_move is not None:
            final_move[best_move] = 1
        return final_move

    def expectimax_decision(self, board: List[List[int]]) -> int:
        args = [(board, move, self.max_depth) for move in range(NUM_DIRECTIONS)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_DIRECTIONS) as executor:
            results = list(executor.map(score_move_for_expectimax, args))
        best_move, best_score = max(results, key=lambda x: x[1])
        return best_move

    def expectimax(self, board: List[List[int]], depth: int, is_player_turn: bool) -> float:
        if depth == 0 or self.is_game_over(board):
            return self.evaluate(board)
        if is_player_turn:
            best_score = float('-inf')
            for move in range(NUM_DIRECTIONS):
                new_board, _, moved = move_board(board, move)
                if moved:
                    score = self.expectimax(new_board, depth - 1, False)
                    best_score = max(best_score, score)
            return best_score
        else:
            empty_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] == 0]
            if not empty_cells:
                return self.evaluate(board)
            total_score = 0.0
            for i, j in empty_cells:
                for value, prob in [(2, 0.9), (4, 0.1)]:
                    new_board = [row[:] for row in board]
                    new_board[i][j] = value
                    score = self.expectimax(new_board, depth - 1, True)
                    total_score += prob * score / len(empty_cells)
            return total_score

    def is_game_over(self, board: List[List[int]]) -> bool:
        for move in range(NUM_DIRECTIONS):
            if is_valid_move_board(board, move):
                return False
        return True

    def evaluate(self, board: List[List[int]]) -> int:
        # Snake pattern para 5x5
        SNAKE_WEIGHTS: list[list[int]] = [
            [2**0,  2**1,  2**2,  0,  0],
            [2**6,  2**5,  2**4,  2**3,  0],
            [2**7,  2**8,  2**9,  2**10, 2**11],
            [2**15, 2**14, 2**13, 2**12, 0],
            [2**16, 2**17, 2**18, 0, 0],
        ]
        h: int = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                h += board[i][j] * SNAKE_WEIGHTS[i][j]
        return h
