from typing import List, Tuple, Optional

# Configuration constants
SEARCH_DEPTH = 6  # For future use when we add minimax

class SimpleOthelloAI:
    def __init__(self):
        pass
    
    def board_to_bitboard(self, board: List[List[int]], player: int) -> Tuple[int, int]:
        """Convert 2D board matrix to bitboards for both players"""
        player_bb = 0
        opponent_bb = 0
        opponent = -player  # if player is 1 (black), opponent is -1 (white)
        
        for row in range(8):
            for col in range(8):
                pos = 1 << (row * 8 + col)
                if board[row][col] == player:
                    player_bb |= pos
                elif board[row][col] == opponent:
                    opponent_bb |= pos
        
        return player_bb, opponent_bb
    
    def bitboard_to_board(self, player_bb: int, opponent_bb: int, player: int) -> List[List[int]]:
        """Convert bitboards back to 2D board matrix"""
        board = [[0 for _ in range(8)] for _ in range(8)]
        opponent = -player
        
        for row in range(8):
            for col in range(8):
                pos = 1 << (row * 8 + col)
                if player_bb & pos:
                    board[row][col] = player
                elif opponent_bb & pos:
                    board[row][col] = opponent
        
        return board
    
    def get_valid_moves_simple(self, board: List[List[int]], player: int) -> List[Tuple[int, int]]:
        """Get valid moves using simple 2D array approach (for now)"""
        valid_moves = []
        opponent = -player
        
        for row in range(8):
            for col in range(8):
                if board[row][col] == 0:  # Empty square
                    if self.is_valid_move(board, row, col, player):
                        valid_moves.append((row, col))
        
        return valid_moves
    
    def is_valid_move(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """Check if a move is valid at given position"""
        if board[row][col] != 0:
            return False
        
        opponent = -player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opponent = False
            
            # Look for opponent pieces in this direction
            while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent:
                found_opponent = True
                r += dr
                c += dc
            
            # If we found opponent pieces and then our piece, it's valid
            if found_opponent and 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
                return True
        
        return False
    
    def bitboard_to_move(self, move_bitboard: int) -> Tuple[int, int]:
        """Convert bitboard position to (row, col) coordinates"""
        if move_bitboard == 0:
            return None
        
        pos = 0
        temp = move_bitboard
        while temp > 1:
            temp >>= 1
            pos += 1
        
        row = pos // 8
        col = pos % 8
        return (row, col)
    
    def print_bitboard(self, bitboard: int, label: str = ""):
        """Debug function to visualize bitboard"""
        if label:
            print(f"{label}:")
        
        for row in range(8):
            row_str = ""
            for col in range(8):
                pos = 1 << (row * 8 + col)
                if bitboard & pos:
                    row_str += "1 "
                else:
                    row_str += "0 "
            print(row_str)
        print()