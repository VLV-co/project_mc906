import numpy as np
from game import Game2048
from agent import ExpectimaxAgent
import time
import os
import csv

MOVE_NAMES = ['MAIN_DIAG_UP', 'SEC_DIAG_UP', 'RIGHT', 'MAIN_DIAG_DOWN', 'SEC_DIAG_DOWN', 'LEFT']

def test(n_games: int = 10) -> None:
    max_depth = 4
    agent = ExpectimaxAgent(max_depth=max_depth)
    record = 0
    scores = []
    max_tiles = []
    game_data = []

    print(f"Testando com ExpectimaxAgent (max_depth={max_depth})")

    # Garante que o diretório de logs existe
    os.makedirs('logs', exist_ok=True)
    csv_path = os.path.join('logs', 'test_log.csv')

    try:
        for i in range(1, n_games + 1):
            game = Game2048()  # Reinicia o jogo
            state = agent.get_state(game)
            moves = []
            move_count = 1

            while not game.game_over:
                final_move = agent.get_action(state, game=game)
                move_idx = int(np.argmax(final_move))
                move_name = MOVE_NAMES[move_idx]
                moves.append(move_name)
                move_count += 1
                moved = game.move(move_idx)
                state = agent.get_state(game)
                if not moved:
                    # Movimento inválido, força término
                    break

            score = game.score
            scores.append(score)
            record = max(record, score)
            max_tile = int(np.max(game.grid))
            max_tiles.append(max_tile)

            # Salva dados da partida
            game_data.append({
                'game_number': i,
                'move_count': move_count - 1,
                'score': score,
                'max_tile': max_tile
            })
            time.sleep(2)

        max_tile = max(max_tiles)
        summary = (
            f"\n--- Test Summary ---\n"
            f"Games Played: {n_games}\n"
            f"Record Score: {record}\n"
            f"Max Tile: {max_tile}\n"
        )
        print(summary)

        # Escreve o CSV
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['game_number', 'move_count', 'score', 'max_tile']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in game_data:
                writer.writerow(row)

    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")

if __name__ == '__main__':
    test(n_games=100)