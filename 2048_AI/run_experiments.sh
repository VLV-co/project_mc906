#!/bin/bash

# Usage: ./run_experiments.sh <board_variant> <size>
# Example: ./run_experiments.sh square 4

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <board_variant> <size>"
    exit 1
fi

BOARD_VARIANT=$1
SIZE=$2

# Run for expectimax agent with all heuristics and depths
for HEURISTIC in empty_cells snake; do
    for DEPTH in 1 2 3 4; do
        python3 main.py \
            -agent expectimax \
            -heuristic $HEURISTIC \
            -depth $DEPTH \
            -n_games 100 \
            -board_variant $BOARD_VARIANT \
            -size $SIZE
    done
done

# Run for random agent (heuristic and depth are not used)
python3 main.py \
    -agent random \
    -n_games 100 \
    -board_variant $BOARD_VARIANT \
    -size $SIZE
