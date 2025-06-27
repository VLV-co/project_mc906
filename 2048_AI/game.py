import pygame
import random
import numpy as np
from typing import Tuple, List

class Game2048:
    def __init__(self, board_variant = 'square', size = 4, animation = False):
        self.board_variant = board_variant
        self.animation = animation
        self.new_tiles = []
        if board_variant == 'square' :
            self.direction = {'right': 0, 'left': 1, 'up': 2, 'down': 3}
        else:
            self.direction =  {'right': 0, 'left': 1, 'up_left': 2, 'down_left': 3, 'up_right': 4, 'down_right': 5}
        self.game_over: bool = False
        self.margin: int = 25
        self.yMarginMultiplier = 1 if board_variant == 'square' else 3**-2 / 2
        self.screen_padding: int = 150
        self.tile_size: int = 100
        self.colors = {
            0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
            8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
            64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
            512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)}
        self.text_colors = {
            2: (119, 110, 101), 4: (119, 110, 101), 8: (249, 246, 242),
            16: (249, 246, 242), 32: (249, 246, 242), 64: (249, 246, 242),
            128: (249, 246, 242), 256: (249, 246, 242), 512: (249, 246, 242),
            1024: (249, 246, 242), 2048: (249, 246, 242)}

        self.board_coordinates = []
        if board_variant != 'hex':
            self.grid_size = size
        else:
            self.grid_size = size * 2 - 1

        self.board_width =  self.grid_size * self.tile_size + (self.grid_size + 1) * self.margin
        self.board_height = self.grid_size * self.tile_size + (self.grid_size - 1) * self.margin * self.yMarginMultiplier + 2 * self.margin 
        self.screen_width = self.board_width + 2 * self.screen_padding
        self.screen_height = self.board_height + 2 * self.screen_padding

        self.new_tiles = []

        # Metrics
        self.moves_without_merge: int = 0
        self.max_tile_history: List[int] = []

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"2048* AI - {board_variant} board")
        self.font_normal = pygame.font.Font("./fonts/GeistMono-SemiBold.otf", 24)
        self.font_label = pygame.font.Font("./fonts/GeistMono-Medium.otf", 20)
        self.font_bold = pygame.font.Font("./fonts/GeistMono-Bold.otf", 24)

        self.reset()
        self._tiles_position()

    def _tiles_position(self):
        self.tiles_position = []
        self.tiles_per_row = [0] * (self.grid_size)

        for i, j in self.board_coordinates:
            self.tiles_per_row[i] += 1

        for row in range(self.grid_size):
            n_tiles = self.tiles_per_row[row]
            row_width = n_tiles * self.tile_size + (n_tiles - 1) * self.margin
            start_x = (self.screen_width - row_width) // 2
            for col in range(n_tiles):
                x = start_x + col * (self.tile_size + self.margin)
                y = self.screen_padding + row * (self.tile_size + self.margin * self.yMarginMultiplier) + self.margin
                self.tiles_position.append((x, y))

    def reset(self) -> None:
        self.board_coordinates.clear()
        self.board = [['x' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.board_init()
        self.board = np.array(self.board, dtype=object)
        self.score: int = 0
        self.game_over: bool = False
        self.game_over_time = None 
        self.moves_without_merge = 0
        self.max_tile_history = []
        self._add_new_tile()
        self._add_new_tile()    

    def board_init(self):
        if self.board_variant == 'triangle':
            for i in range(self.grid_size):
                for j in range(i + 1):
                    self.board[i][j] = 0
                    self.board_coordinates.append((i, j))
        elif self.board_variant == 'hex':
            size = (self.grid_size + 1) // 2
            for i in range(size, self.grid_size):
                for j in range(i):
                    self.board[i - size][j] = 0
                    self.board[size * 3 - 2 - i][self.grid_size - j - 1] = 0
                    self.board_coordinates.append((i - size, j))
                    self.board_coordinates.append((size * 3 - 2 - i, self.grid_size - j - 1))

            for k in range(self.grid_size):
                self.board[size - 1][k] = 0
                self.board_coordinates.append((size - 1, k))

            self.board_coordinates = sorted(self.board_coordinates, key=lambda pos: (pos[0], pos[1]))
        else:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    self.board[i][j] = 0
                    self.board_coordinates.append((i, j))

    def _add_new_tile(self) -> None:
        empty_cells = [(i, j) for i, j in self.board_coordinates if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4
            if self.animation: self.new_tiles.append((i, j, pygame.time.get_ticks()))        

    def _move_tiles(self, direction, board):
        original_board = board.copy()
            
        if direction == self.direction['left']:
            score = self.merge_left(board)
        elif direction == self.direction['right']:
            board = np.fliplr(board)
            score = self.merge_left(board) 
            board = np.fliplr(board)
        elif self.board_variant == 'square':
            if direction == self.direction['up']:
                board = np.rot90(board)
                score = self.merge_left(board) 
                board = np.rot90(board, k=3)
            elif direction == self.direction['down']:
                board = np.rot90(board, k=3)
                score = self.merge_left(board) 
                board = np.rot90(board)
        else:
            if direction == self.direction['up_left']:
                self.mirror(board)
                board = np.rot90(board)
                score = self.merge_left(board) 
                board = np.rot90(board, k=3)
                self.mirror(board)
            elif direction == self.direction['down_left']:
                self.mirror(board)
                board = np.rot90(board, k=3)
                score = self.merge_left(board) 
                board = np.rot90(board)
                self.mirror(board)
            elif direction == self.direction['up_right']:
                board = np.rot90(board)
                score = self.merge_left(board) 
                board = np.rot90(board, k=3)
            elif direction == self.direction['down_right']:
                board = np.rot90(board, k=3)
                score = self.merge_left(board) 
                board = np.rot90(board)

        return score, not np.array_equal(original_board, board)  

    def merge_left(self,board):
        for i in range(len(board[0])): 
            row, score = self.merge(board[i])

            while len(row) < len([v for v in board[i] if v != 'x']):
                row.append(0)

            result = []
            valid_idx = 0
            for v in board[i]:
                if v == 'x':
                    result.append('x')
                else:
                    result.append(row[valid_idx])
                    valid_idx += 1
            board[i] = result

        return score

    def mirror(self, board):
        for i, row in enumerate(board):
            values = [v for v in row if v != 'x'][::-1]
            indices = [j for j, v in enumerate(row) if v != 'x']

            for j, val in zip(indices, values):
                row[j] = val

    def merge(self, row: List[int]) -> Tuple[List[int], int]:
        new_row = [tile for tile in row if tile != 0 and tile != 'x']
        score = 0
        i = 0
        while i < len(new_row) - 1:
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                score += new_row[i]
                del new_row[i + 1]
            else:
                i += 1

        return new_row, score

    def get_max_tile(self) -> int:
        return int(np.max(self.board[self.board != 'x']))
    
    def _has_empty_cells(self) -> int:
        for row in self.board:
            if 0 in row:
                return True
        return False

    def _has_valid_moves(self):
        for direction in self.direction.values():
            if self._is_valid_move(direction):
                return True
        return False

    def _is_valid_move(self, direction) -> bool:
        temp_board = self.board.copy()
        self._move_tiles(direction, temp_board)
        return not np.array_equal(temp_board, self.board)
    
    def play_step(self, direction) -> Tuple[bool, int]:        
        score_increase, moved = self._move_tiles(direction, self.board)
        if moved:
            self.score += score_increase
            self.new_tiles.clear()
            self._add_new_tile()
            self.draw_board()

        if not self._has_empty_cells():
            if not self._has_valid_moves():
                self.game_over = True
                self.game_over_time = pygame.time.get_ticks()
        else:
            self.game_over = False


        return self.game_over, self.score
    
    def _draw_tile(self, surface, value, i, scale=1.0):
        size = int(self.tile_size * scale)
        offset = (self.tile_size - size) // 2
        x = self.tiles_position[i][0] + offset
        y = self.tiles_position[i][1] + offset

        pygame.draw.circle(surface, self.colors.get(value, (237, 160, 20)), (x + size // 2, y + size // 2), size // 2)
        if value != 0 and scale > 0.7:
            font_size = 36 if value < 1000 else 24
            font = pygame.font.Font("./fonts/GeistMono-SemiBold.otf", font_size)
            text = font.render(str(value), True, self.text_colors.get(value, (255, 255, 255)))
            text_rect = text.get_rect(center=(x + size // 2, y + size // 2))
            surface.blit(text, text_rect)

    def draw_board(self):
        now = pygame.time.get_ticks()

        self.screen.fill((250, 248, 239))
        pygame.draw.rect(self.screen, (187, 173, 160), (self.screen_padding, self.screen_padding, self.board_width , self.board_height), 0, 10)

        for n, (i, j) in enumerate(self.board_coordinates):
            scale = 1.0
            for (a, b, start_time) in self.new_tiles:
                if a == i and b == j:
                    elapsed = now - start_time
                    duration = 200
                    progress = min(elapsed / duration, 1.0)
                    scale = 0.5 + 0.5 * (1 - (1 - progress) ** 2)
                    break

            self._draw_tile(self.screen, self.board[i][j], n, scale)

        rect_width, rect_height = 130, 80
        score_rect = pygame.Rect((self.screen_width - rect_width) // 2, 10, rect_width, rect_height)
        pygame.draw.rect(self.screen, (119, 110, 101), score_rect, border_radius=16)

        score_label = self.font_label.render("Score", True, (255, 255, 255))
        score_value = self.font_bold.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_label, (score_rect.centerx - score_label.get_width() // 2, score_rect.y + 10))
        self.screen.blit(score_value, (score_rect.centerx - score_value.get_width() // 2, score_rect.y + 40))

        pygame.display.flip()

if __name__ == '__main__':
    game = Game2048('triangle')  # Change to 'square', 'triangle', or 'hex' as needed

    while True:
        if not game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        game.reset()
                    elif event.key in [pygame.K_w]:
                        game.play_step(2)
                    elif event.key in [pygame.K_e]:
                        game.play_step(4)
                    elif event.key in [pygame.K_d]:
                        game.play_step(0)
                    elif event.key in [pygame.K_x]:
                        game.play_step(3)
                    elif event.key in [pygame.K_z]:
                        game.play_step(5)
                    elif event.key in [pygame.K_a]:
                        game.play_step(1)
                    elif event.key in [pygame.K_s]:
                        game.play_step(3)

        game.draw_board()
        if game.game_over:
            print(f'Game Over! Score: {game.score}')
            if pygame.time.get_ticks() - game.game_over_time > 5000:
                game.reset()
        pygame.time.Clock().tick(60) 