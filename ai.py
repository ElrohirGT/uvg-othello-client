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
        """Get valid moves using bitboard-based validation"""
        valid_moves = []
        player_bb, opponent_bb = self.board_to_bitboard(board, player)

        for row in range(8):
            for col in range(8):
                move_pos = row * 8 + col
                bit = 1 << move_pos
                if (player_bb | opponent_bb) & bit:
                    continue  # Not empty

                if self.is_valid_move_bitboard(move_pos, player_bb, opponent_bb):
                    valid_moves.append((row, col))

        return valid_moves


    def is_valid_move_bitboard(self, move_pos: int, player_bb: int, opponent_bb: int) -> bool:
        """Check if placing a piece at move_pos is valid using bitboards"""
        if (player_bb | opponent_bb) & (1 << move_pos):
            return False  # Square is not empty

        DIRECTIONS = [
            1, -1,        # E, W
            8, -8,        # S, N
            9, -9,        # SE, NW
            7, -7         # SW, NE
        ]

        MASK_LEFT = 0xfefefefefefefefe
        MASK_RIGHT = 0x7f7f7f7f7f7f7f7f

        for direction in DIRECTIONS:
            mask = 0xFFFFFFFFFFFFFFFF
            pos = move_pos
            found_opponent = False

            while True:
                if direction == -1 and pos % 8 == 0: break
                if direction == 1 and pos % 8 == 7: break
                if direction == -9 and (pos % 8 == 0 or pos < 8): break
                if direction == -8 and pos < 8: break
                if direction == -7 and (pos % 8 == 7 or pos < 8): break
                if direction == 7 and (pos % 8 == 0 or pos > 55): break
                if direction == 8 and pos > 55: break
                if direction == 9 and (pos % 8 == 7 or pos > 55): break

                pos += direction
                if pos < 0 or pos > 63:
                    break

                bit = 1 << pos
                if opponent_bb & bit:
                    found_opponent = True
                    continue
                elif player_bb & bit:
                    if found_opponent:
                        return True
                    break
                else:
                    break

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

    def evaluate_position(self, board: Tuple[int, int], player: int) -> float:
        player_bb, opponent_bb = self.board_to_bitboard(board, player)
        total_discs = bin(player_bb | opponent_bb).count("1")
        
        def count_bits(bb: int) -> int:
            return bin(bb).count("1")

        # Phase detection
        if total_discs <= 20:
            phase = 'early'
        elif total_discs <= 54:
            phase = 'mid'
        else:
            phase = 'late'

        # Mobility
        my_moves = len(self.get_valid_moves_simple(board, player))
        opp_moves = len(self.get_valid_moves_simple(board, -player))
        mobility = 0
        if my_moves + opp_moves > 0:
            mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

        # Corners
        corners = [0, 7, 56, 63]
        corner_mask = sum(1 << i for i in corners)
        player_corners = count_bits(player_bb & corner_mask)
        opponent_corners = count_bits(opponent_bb & corner_mask)
        corner_score = 25 * (player_corners - opponent_corners)

        # Disc differential (only matter late)
        disc_score = 0
        if phase == 'late':
            player_discs = count_bits(player_bb)
            opponent_discs = count_bits(opponent_bb)
            disc_score = 100 * (player_discs - opponent_discs) / (player_discs + opponent_discs)

        # Final evaluation (tune weights as needed)
        if phase == 'early':
            return mobility + corner_score
        elif phase == 'mid':
            return mobility + 1.5 * corner_score
        else:  # late
            return mobility + 2 * corner_score + disc_score