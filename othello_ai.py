import random
from typing import List, Tuple, Optional
from ai import SimpleOthelloAI


DIRECTIONS = [
    (-1, -1),  # UP-LEFT
    (-1, 0),   # UP
    (-1, 1),   # UP-RIGHT
    (0, -1),   # LEFT
    (0, 1),    # RIGHT
    (1, -1),   # DOWN-LEFT
    (1, 0),    # DOWN
    (1, 1)     # DOWN-RIGHT
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
    symbol_map = {
        1: '●',   # Black
        -1: '○',  # White
        0: '.'    # Empty
    }
    for row in board:
        print(" ".join(symbol_map.get(cell, '?') for cell in row), flush=True)

SEARCH_DEPTH = 6  # Adjust this value to control AI strength vs speed

# Your original function with simple AI
def ai_move(board, player):
    
    if player == 1: 
        print("WHITE", flush=True)
    else :
        print("BLACK", flush=True)

    """Main AI function - now just picks random valid move"""

    print("=============================", flush=True)
    print_othello_board(board)

    ai = SimpleOthelloAI()
    
    print(f"AI move called for player {player}")
    
    # Convert board to bitboards (for testing)
    player_bb, opponent_bb = ai.board_to_bitboard(board, player)
    print(f"Player bitboard: {bin(player_bb)}")
    print(f"Opponent bitboard: {bin(opponent_bb)}")
    
    # For now, use simple 2D approach to find valid moves
    valid_moves = ai.get_valid_moves_simple(board, player)
    print(ai.evaluate_position(board, player), flush=True)
    print(f"Valid moves found: {valid_moves}")
    
    if valid_moves:
        chosen_move = random.choice(valid_moves)
        print(f"Chosen move: {chosen_move}")
        return chosen_move
    
    print("No valid moves found, returning None", flush=True)
    return None