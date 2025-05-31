import random
from typing import List, Tuple, Optional
from ai import SimpleOthelloAI
import time
import subprocess


DIRECTIONS = [
    (-1, -1),  # UP-LEFT
    (-1, 0),  # UP
    (-1, 1),  # UP-RIGHT
    (0, -1),  # LEFT
    (0, 1),  # RIGHT
    (1, -1),  # DOWN-LEFT
    (1, 0),  # DOWN
    (1, 1),  # DOWN-RIGHT
]


def in_bounds(x, y):
    return 0 <= x < 8 and 0 <= y < 8


def valid_movements(board, player):
    opponent = -player
    valid_moves = []

    for x in range(8):
        for y in range(8):
            if board[x][y] != 0:
                continue

            for dx, dy in DIRECTIONS:
                i, j = x + dx, y + dy
                found_opponent = False

                while in_bounds(i, j) and board[i][j] == opponent:
                    i += dx
                    j += dy
                    found_opponent = True

                if found_opponent and in_bounds(i, j) and board[i][j] == player:
                    valid_moves.append((x, y))
                    break

    return valid_moves


def print_othello_board(board):
    symbol_map = {1: "●", -1: "○", 0: "."}  # Black  # White  # Empty
    for row in board:
        print(" ".join(symbol_map.get(cell, "?") for cell in row), flush=True)


SEARCH_DEPTH = 6  # Adjust this value to control AI strength vs speed
ai = SimpleOthelloAI()
ai_brain = None  # Holds the process handler

max_time = 0


# Your original function with simple AI
def ai_move(board, player):
    global max_time, ai_brain
    start_time = time.time()

    if player == 1:
        print("WHITE", flush=True)
    else:
        print("BLACK", flush=True)

    """Main AI function - now just picks random valid move"""

    print("=============================", flush=True)
    print_othello_board(board)

    print(f"AI move called for player {player}", flush=True)

    # Convert board to bitboards (for testing)
    player_bb, opponent_bb = ai.board_to_bitboard(board, player)
    # print(f"Player bitboard: {bin(player_bb)}")
    # print(f"Opponent bitboard: {bin(opponent_bb)}")

    # For now, use simple 2D approach to find valid moves
    # valid_moves = ai.get_valid_moves_simple(player_bb, opponent_bb)
    # print(ai.evaluate_position(player_bb, opponent_bb), flush=True)
    # print(f"Valid moves found: {valid_moves}")

    # best_move = ai.select_best_move(player_bb, opponent_bb, 6)
    if ai_brain == None:
        print("Creating ai_brain...", flush=True)
        ai_brain = subprocess.Popen(
            ["./ia", "-s=7"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        print("ai_brain created!", flush=True)

    ai_brain_command = f"{str(player_bb)} {str(opponent_bb)}\n"
    print("Sending command:", ai_brain_command, flush=True)
    ai_brain.stdin.write(ai_brain_command.encode("utf-8"))
    ai_brain.stdin.flush()
    print("Awaiting response...")
    response = ai_brain.stdout.readline().decode("utf-8")
    print("Brain response:", response, flush=True)
    best_move = [int(x) for x in response.split(" ")]
    # result = subprocess.run(
    #     ["./ia", "-pBB=" + str(player_bb), "-opBB=" + str(opponent_bb), "-s=10"],
    #     capture_output=True,
    #     text=True,
    # )
    # print(f"The result is: {result}")

    # best_move = [int(x) for x in result.stdout.split(" ")]
    print(f"The best move is: {best_move}", flush=True)

    took = time.time() - start_time
    print(f"Took: {took}", flush=True)

    if max_time < took:
        print("NEW MAX", flush=True)
        max_time = took

    return best_move
