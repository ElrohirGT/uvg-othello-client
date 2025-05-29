import random
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
    
    def get_valid_moves_simple(self, player_bb: int, opponent_bb: int) -> List[Tuple[int, int]]:
        """Get valid moves using bitboard-based validation"""
        valid_moves = []
        occupied = player_bb | opponent_bb

        for pos in range(64):
            bit = 1 << pos
            if occupied & bit:
                continue  # Not empty

            if self.is_valid_move_bitboard(pos, player_bb, opponent_bb):
                row, col = divmod(pos, 8)
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

    def apply_move(self, move_pos: int, player_bb: int, opponent_bb: int) -> Tuple[int, int]:
        """Apply a move and return updated (player_bb, opponent_bb)"""
        flipped = 0
        DIRECTIONS = [
            1, -1, 8, -8, 9, -9, 7, -7
        ]

        for direction in DIRECTIONS:
            pos = move_pos
            captured = 0

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
                    captured |= bit
                elif player_bb & bit:
                    flipped |= captured
                    break
                else:
                    break

        move_bit = 1 << move_pos
        new_player_bb = player_bb | move_bit | flipped
        new_opponent_bb = opponent_bb & ~flipped
        return new_player_bb, new_opponent_bb

    def minimax(self, player_bb: int, opponent_bb: int, depth: int, maximizing_player: bool) -> float:
        """Negamax-style minimax with depth limit"""
        if depth == 0:
            return self.evaluate_position(player_bb, opponent_bb)

        valid_moves = self.get_valid_moves_simple(player_bb, opponent_bb)

        if not valid_moves:
            # Try passing the turn if opponent still has moves
            if self.get_valid_moves_simple(opponent_bb, player_bb):
                return -self.minimax(opponent_bb, player_bb, depth, not maximizing_player)
            else:
                # No valid moves for either: game over
                return self.evaluate_position(player_bb, opponent_bb)

        best_score = float('-inf')
        for row, col in valid_moves:
            move_pos = row * 8 + col
            new_player_bb, new_opponent_bb = self.apply_move(move_pos, player_bb, opponent_bb)
            score = -self.minimax(new_opponent_bb, new_player_bb, depth - 1, not maximizing_player)
            best_score = max(best_score, score)

        return best_score

    def select_best_move(self, player_bb: int, opponent_bb: int, depth: int) -> Tuple[int, int]:
        """Find best move using minimax with guaranteed fallback (random among valid moves)"""
        valid_moves = self.get_valid_moves_simple(player_bb, opponent_bb)

        if not valid_moves:
            # Fallback: choose randomly from all empty squares (unlikely, but guarantees a move)
            empty_squares = [(r, c) for r in range(8) for c in range(8)
                             if not ((player_bb | opponent_bb) & (1 << (r * 8 + c)))]
            return random.choice(empty_squares) if empty_squares else (0, 0)  # Safety fallback

        best_score = float('-inf')
        best_moves = []

        for row, col in valid_moves:
            move_pos = row * 8 + col
            new_player_bb, new_opponent_bb = self.apply_move(move_pos, player_bb, opponent_bb)
            score = -self.minimax(new_opponent_bb, new_player_bb, depth - 1, False)

            if score > best_score:
                best_score = score
                best_moves = [(row, col)]
            elif score == best_score:
                best_moves.append((row, col))

        return random.choice(best_moves)

    def evaluate_position(self, player_bb: int, opponent_bb: int) -> float:
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
        my_moves = len(self.get_valid_moves_simple(player_bb, opponent_bb))
        opp_moves = len(self.get_valid_moves_simple(opponent_bb, player_bb))
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