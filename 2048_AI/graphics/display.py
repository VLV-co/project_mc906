import pygame
import numpy as np
from typing import Tuple, Optional
from ..game.game_logic import Game2048

class GameDisplay:
    def __init__(
        self,
        cell_size: int = 100,
        padding: int = 10,
        background_color: Tuple[int, int, int] = (187, 173, 160),
        empty_cell_color: Tuple[int, int, int] = (205, 193, 180),
        text_color: Tuple[int, int, int] = (119, 110, 101),
        font_size: int = 36
    ) -> None:
        self.cell_size = cell_size
        self.padding = padding
        self.background_color = background_color
        self.empty_cell_color = empty_cell_color
        self.text_color = text_color
        self.font_size = font_size
        
        # Calculate window size
        self.window_size = (
            cell_size * 4 + padding * 5,
            cell_size * 4 + padding * 5 + 100  # Extra space for score
        )
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("2048 AI")
        
        # Initialize font
        self.font = pygame.font.Font(None, font_size)
        self.score_font = pygame.font.Font(None, font_size // 2)
        
        # Color mapping for different tile values
        self.tile_colors = {
            0: empty_cell_color,
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

    def draw_board(self, game: Game2048) -> None:
        """Draw the game board."""
        self.screen.fill(self.background_color)
        
        # Draw cells
        for i in range(4):
            for j in range(4):
                value = game.board[i][j]
                color = self.tile_colors.get(value, (237, 194, 46))
                
                # Calculate cell position
                x = self.padding + j * (self.cell_size + self.padding)
                y = self.padding + i * (self.cell_size + self.padding)
                
                # Draw cell
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x, y, self.cell_size, self.cell_size),
                    border_radius=6
                )
                
                # Draw value if not empty
                if value != 0:
                    text = self.font.render(str(value), True, self.text_color)
                    text_rect = text.get_rect(center=(x + self.cell_size/2, y + self.cell_size/2))
                    self.screen.blit(text, text_rect)
        
        # Draw score
        score_text = self.score_font.render(f"Score: {game.get_score()}", True, self.text_color)
        self.screen.blit(score_text, (self.padding, self.window_size[1] - 40))
        
        pygame.display.flip()

    def handle_events(self) -> Optional[int]:
        """Handle Pygame events and return the action if any."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return 0
                elif event.key == pygame.K_RIGHT:
                    return 1
                elif event.key == pygame.K_DOWN:
                    return 2
                elif event.key == pygame.K_LEFT:
                    return 3
        return None

    def close(self) -> None:
        """Close the Pygame window."""
        pygame.quit()

def play_game() -> None:
    """Play the game manually."""
    game = Game2048()
    display = GameDisplay()
    
    while True:
        display.draw_board(game)
        action = display.handle_events()
        
        if action is None:
            break
            
        if game.move(action):
            game.add_new_tile()
            
            if game.is_game_over():
                print(f"Game Over! Final Score: {game.get_score()}")
                break
    
    display.close() 