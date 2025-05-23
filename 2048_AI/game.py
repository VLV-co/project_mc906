import math
import pygame
import random
import sys

# Constantes
GRID_SIZE = 5
TILE_SIZE = 100
GAP = 20
BOARD_WIDTH = GRID_SIZE * (TILE_SIZE + GAP) + GAP
BOARD_HEIGHT = GRID_SIZE * (TILE_SIZE + GAP) + GAP

PADDING = 100
WIDTH = HEIGHT = BOARD_WIDTH + PADDING * 2

# Cores
BACKGROUND_COLOR = (187, 173, 160)
EMPTY_TILE_COLOR = (205, 193, 180)
TILE_COLORS = {
    0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)}
TEXT_COLORS = {
    2: (119, 110, 101), 4: (119, 110, 101), 8: (249, 246, 242),
    16: (249, 246, 242), 32: (249, 246, 242), 64: (249, 246, 242),
    128: (249, 246, 242), 256: (249, 246, 242), 512: (249, 246, 242),
    1024: (249, 246, 242), 2048: (249, 246, 242)}

# Inicialização do pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048")
clock = pygame.time.Clock()

# Fontes
font_label = pygame.font.Font("./fonts/GeistMono-Medium.otf", 20)
font_normal = pygame.font.Font("./fonts/GeistMono-SemiBold.otf", 24)
font_bold = pygame.font.Font("./fonts/GeistMono-Bold.otf", 24)

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

class Game2048:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.score = 0
        self.game_over = False
        self.won = False
        self.new_tiles = []  # Lista para armazenar novos blocos animados
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty_cells = [(i, j) for i, j in GRID_COORDENATES if self.grid[i][j] == 0]

        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4
            self.new_tiles.append((i, j, pygame.time.get_ticks()))

    def move(self, direction):
        if self.game_over:
            return False

        moved = False
        self.new_tiles.clear()

        if direction == 0: # main diagonal up
            for k in range(GRID_SIZE):
                coord = MAIN_DIAGONAL[k]
                transversal = [self.grid[i][j] for (i,j) in coord]
                new_transversal, add_score = self._merge(transversal)
                for l in range(len(transversal)):
                    if transversal[l] != new_transversal[l]:
                        moved = True
                        i, j = coord[l]
                        self.grid[i][j] = new_transversal[l]
                self.score += add_score
        elif direction == 1: # secondary diagonal up
            for k in range(GRID_SIZE):
                coord = SECONDARY_DIAGONAL[k]
                transversal = [self.grid[i][j] for (i,j) in coord]
                new_transversal, add_score = self._merge(transversal)
                for l in range(len(transversal)):
                    if transversal[l] != new_transversal[l]:
                        moved = True
                        i, j = coord[l]
                        self.grid[i][j] = new_transversal[l]
                self.score += add_score
        elif direction == 2: # right
            for k in range(GRID_SIZE):
                row = [self.grid[i][j] for (i,j) in GRID_COORDENATES if i == k]
                row.reverse()
                new_row, add_score = self._merge(row)
                new_row.reverse()
                for j in range(len(row)):
                    if self.grid[k][j] != new_row[j]:
                        moved = True
                        self.grid[k][j] = new_row[j]
                self.score += add_score
        elif direction == 3: # main diagonal down
            for k in range(GRID_SIZE):
                coord = MAIN_DIAGONAL[k]
                transversal = [self.grid[i][j] for (i,j) in coord]
                transversal.reverse()
                new_transversal, add_score = self._merge(transversal)
                new_transversal.reverse()
                for l in range(len(transversal)):
                    i, j = coord[l]
                    if self.grid[i][j] != new_transversal[l]:
                        moved = True
                        self.grid[i][j] = new_transversal[l]
                self.score += add_score
        elif direction == 4: # secondary diagonal down
            for k in range(GRID_SIZE):
                coord = SECONDARY_DIAGONAL[k]
                transversal = [self.grid[i][j] for (i,j) in coord]
                transversal.reverse()
                new_transversal, add_score = self._merge(transversal)
                new_transversal.reverse()
                for l in range(len(transversal)):
                    i, j = coord[l]
                    if self.grid[i][j] != new_transversal[l]:
                        moved = True
                        self.grid[i][j] = new_transversal[l]
                self.score += add_score
        elif direction == 5: # left
            for k in range(GRID_SIZE):
                row = [self.grid[i][j] for (i,j) in GRID_COORDENATES if i == k]
                new_row, add_score = self._merge(row)
                for j in range(len(row)):
                    if self.grid[k][j] != new_row[j]:
                        moved = True
                        self.grid[k][j] = new_row[j]
                self.score += add_score

        if moved:
            self.add_random_tile()
            if self._check_win():
                self.won = True
            if self._check_game_over():
                self.game_over = True
        return moved

    def _merge(self, line):
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

    def _check_win(self):
        return any(self.grid[i][j] >= 2048 for i in range(GRID_SIZE) for j in range(GRID_SIZE))

    def _check_game_over(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == 0:
                    return False
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return False
        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return False
        return True

def draw_tile(surface, value, x, y, scale=1.0):
    size = int(TILE_SIZE * scale)
    offset = (TILE_SIZE - size) // 2
    x += offset
    y += offset

    pygame.draw.circle(surface, TILE_COLORS.get(value, (237, 160, 20)), (x + size // 2, y + size // 2), size // 2)
    if value != 0 and scale > 0.7:
        font_size = 36 if value < 1000 else 24
        font = pygame.font.Font("./fonts/GeistMono-SemiBold.otf", font_size)
        text_color = TEXT_COLORS.get(value, (249, 246, 242))
        text = font.render(str(value), True, text_color)
        text_rect = text.get_rect(center=(x + size // 2, y + size // 2))
        surface.blit(text, text_rect)

def draw_board(surface, game, tiles_positions):
    pygame.draw.rect(surface, BACKGROUND_COLOR, (PADDING, PADDING, BOARD_WIDTH, BOARD_HEIGHT), 0, 10)
    now = pygame.time.get_ticks()
    for n, (i, j) in enumerate(GRID_COORDENATES):
        x, y = tiles_positions[n]

        scale = 1.0
        for (r, c, start_time) in game.new_tiles:
            if r == i and c == j:
                elapsed = now - start_time
                duration = 200
                progress = min(elapsed / duration, 1.0)
                scale = 0.5 + 0.5 * (1 - (1 - progress) ** 2)
                break

        draw_tile(surface, game.grid[i][j], x + PADDING/2, y + PADDING/2, scale)

    # Score e overlay
    rect_width, rect_height = 130, 80
    score_rect = pygame.Rect((WIDTH - rect_width) // 2, 10, rect_width, rect_height)
    pygame.draw.rect(screen, (119, 110, 101), score_rect, border_radius=16)

    score_label = font_label.render("Score", True, (255, 255, 255))
    score_value = font_bold.render(str(game.score), True, (255, 255, 255))
    screen.blit(score_label, (score_rect.centerx - score_label.get_width() // 2, score_rect.y + 10))
    screen.blit(score_value, (score_rect.centerx - score_value.get_width() // 2, score_rect.y + 40))

    if game.won or game.game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 150))
        surface.blit(overlay, (0, 0))
        font = pygame.font.SysFont("Arial", 48, bold=True)
        if game.won:
            text = font.render("You Win!", True, (119, 110, 101))
        else:
            text = font.render("Game Over!", True, (119, 110, 101))
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        surface.blit(text, text_rect)
        font = pygame.font.SysFont("Arial", 24, bold=True)
        if game.won:
            msg = font_bold.render("Continue playing? Press 'C'", True, (119, 110, 101))
        else:
            msg = font.render("Press 'R' to retry", True, (119, 110, 101))
        msg_rect = msg.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20))
        surface.blit(msg, msg_rect)

def tiles_position(centerX, centerY):
    positions = [(centerX, centerY)]

    for j in range(1, 3):
        for i in range(6):
            angle = i * 60 * (math.pi / 180)
            x = round(centerX + math.cos(angle) * (TILE_SIZE+GAP) * j, 3)
            y = round(centerY + math.sin(angle) * (TILE_SIZE+GAP) * j, 3)
            positions.append((x, y))

    for k in range(6):
        angle = (30 + k * 60) * (math.pi / 180)
        x = round(centerX + math.cos(angle) * (TILE_SIZE+GAP) * math.sqrt(3), 3)
        y = round(centerY + math.sin(angle) * (TILE_SIZE+GAP) * math.sqrt(3), 3)
        positions.append((x, y))

    return sorted(positions, key=lambda pos: (pos[1], pos[0]))

def main():
    game = Game2048()
    running = True
    tiles_positions = tiles_position(BOARD_WIDTH/2, BOARD_HEIGHT/2)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Game2048()
                if game.won and event.key == pygame.K_c:
                    game.won = False
                if not game.game_over or (game.won and not game.game_over):
                    if event.key in [pygame.K_w]:
                        game.move(0)
                    elif event.key in [pygame.K_e]:
                        game.move(1)
                    elif event.key in [pygame.K_d]:
                        game.move(2)
                    elif event.key in [pygame.K_x]:
                        game.move(3)
                    elif event.key in [pygame.K_z]:
                        game.move(4)
                    elif event.key in [pygame.K_a]:
                        game.move(5)

        screen.fill((250, 248, 239))
        draw_board(screen, game, tiles_positions)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
