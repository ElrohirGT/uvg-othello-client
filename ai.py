import random
from typing import List, Tuple, Optional

# Configuration constants
SEARCH_DEPTH = 6  # Default search depth for minimax

class SimpleOthelloAI:
    def __init__(self):
        self.branch_count = 0  # Track how many branches are evaluated during search

    def board_to_bitboard(self, board: List[List[int]], player: int) -> Tuple[int, int]:
        """
        Convert a 2D board into two 64-bit integers (bitboards),
        one for the current player and one for the opponent.
        """
        player_bb = 0
        opponent_bb = 0
        opponent = -player
        
        for row in range(8):
            for col in range(8):
                pos = 1 << (row * 8 + col)
                if board[row][col] == player:
                    player_bb |= pos
                elif board[row][col] == opponent:
                    opponent_bb |= pos
        
        return player_bb, opponent_bb

    def bitboard_to_board(self, player_bb: int, opponent_bb: int, player: int) -> List[List[int]]:
        """
        Convert bitboards back to a 2D board representation.
        Used for visualization or fallback.
        """
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
        """
        Iterate over all board positions to find valid moves
        for the current player based on the bitboard state.
        """
        valid_moves = []
        occupied = player_bb | opponent_bb

        for pos in range(64):
            bit = 1 << pos
            if occupied & bit:
                continue

            if self.is_valid_move_bitboard(pos, player_bb, opponent_bb):
                row, col = divmod(pos, 8)
                valid_moves.append((row, col))

        return valid_moves

    def is_valid_move_bitboard(self, move_pos: int, player_bb: int, opponent_bb: int) -> bool:
        """
        Check whether placing a piece at a given position is a valid move.
        Looks in all 8 directions to find if at least one opponent piece is flanked.
        """
        if (player_bb | opponent_bb) & (1 << move_pos):
            return False  # Already occupied

        DIRECTIONS = [1, -1, 8, -8, 9, -9, 7, -7]  # 8 surrounding directions

        for direction in DIRECTIONS:
            pos = move_pos
            found_opponent = False

            while True:
                # Break if we cross board boundaries
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
                        return True  # Found a bracketed line
                    break
                else:
                    break

        return False  # No direction formed a valid capture

    def bitboard_to_move(self, move_bitboard: int) -> Tuple[int, int]:
        """
        Convert a bitboard with a single set bit into (row, col) coordinates.
        """
        if move_bitboard == 0:
            return None
        
        pos = 0
        while move_bitboard > 1:
            move_bitboard >>= 1
            pos += 1

        return divmod(pos, 8)

    def print_bitboard(self, bitboard: int, label: str = ""):
        """
        Print the bitboard in a human-readable 8x8 format.
        Used for debugging.
        """
        if label:
            print(f"{label}:")
        
        for row in range(8):
            row_str = ""
            for col in range(8):
                pos = 1 << (row * 8 + col)
                row_str += "1 " if bitboard & pos else "0 "
            print(row_str)
        print()

    def apply_move(self, move_pos: int, player_bb: int, opponent_bb: int) -> Tuple[int, int]:
        """
        Apply a move at `move_pos` to flip appropriate discs.
        Returns updated bitboards after move execution.
        """
        flipped = 0
        DIRECTIONS = [1, -1, 8, -8, 9, -9, 7, -7]

        for direction in DIRECTIONS:
            pos = move_pos
            captured = 0

            while True:
                # Boundary checks
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
        return player_bb | move_bit | flipped, opponent_bb & ~flipped

    def minimax(self, player_bb: int, opponent_bb: int, depth: int, maximizing_player: bool) -> float:
        """
        Basic minimax implementation without pruning.
        Uses negamax convention (scores are negated across recursion).
        """
        if depth == 0:
            return self.evaluate_position(player_bb, opponent_bb)

        valid_moves = self.get_valid_moves_simple(player_bb, opponent_bb)

        if not valid_moves:
            if self.get_valid_moves_simple(opponent_bb, player_bb):
                return -self.minimax(opponent_bb, player_bb, depth, not maximizing_player)
            else:
                return self.evaluate_position(player_bb, opponent_bb)

        best_score = float('-inf')
        for row, col in valid_moves:
            self.branch_count += 1
            move_pos = row * 8 + col
            new_player_bb, new_opponent_bb = self.apply_move(move_pos, player_bb, opponent_bb)
            score = -self.minimax(new_opponent_bb, new_player_bb, depth - 1, not maximizing_player)
            best_score = max(best_score, score)

        return best_score

    def minimax_alpha_beta(self, player_bb, opponent_bb, depth, alpha, beta, maximizing_player) -> float:
        """
        Minimax implementation with alpha-beta pruning.
        Prunes branches that can't improve the outcome.
        """
        if depth == 0:
            return self.evaluate_position(player_bb, opponent_bb)

        valid_moves = self.get_valid_moves_simple(player_bb, opponent_bb)

        if not valid_moves:
            if self.get_valid_moves_simple(opponent_bb, player_bb):
                return -self.minimax_alpha_beta(opponent_bb, player_bb, depth, -beta, -alpha, not maximizing_player)
            else:
                return self.evaluate_position(player_bb, opponent_bb)

        best_score = float('-inf')
        for row, col in valid_moves:
            self.branch_count += 1
            move_pos = row * 8 + col
            new_player_bb, new_opponent_bb = self.apply_move(move_pos, player_bb, opponent_bb)
            score = -self.minimax_alpha_beta(new_opponent_bb, new_player_bb, depth - 1, -beta, -alpha, not maximizing_player)
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # Beta cutoff

        return best_score

    def select_best_move(self, player_bb: int, opponent_bb: int, depth: int) -> Tuple[int, int]:
        """
        Choose the best move by searching game tree using minimax with alpha-beta pruning.
        Randomly breaks ties between equally good moves.
        """
        self.branch_count = 0
        valid_moves = self.get_valid_moves_simple(player_bb, opponent_bb)

        if not valid_moves:
            # Fallback: pick random empty square if no valid moves (unusual case)
            empty_squares = [(r, c) for r in range(8) for c in range(8)
                             if not ((player_bb | opponent_bb) & (1 << (r * 8 + c)))]
            return random.choice(empty_squares) if empty_squares else (0, 0)

        best_score = float('-inf')
        best_moves = []

        for row, col in valid_moves:
            move_pos = row * 8 + col
            new_player_bb, new_opponent_bb = self.apply_move(move_pos, player_bb, opponent_bb)
            score = -self.minimax_alpha_beta(new_opponent_bb, new_player_bb, depth - 1, float('-inf'), float('inf'), False)
            # Uncommend to use normal minimax function without prunning
            # score = -self.minimax(new_opponent_bb, new_player_bb, depth - 1, False)

            if score > best_score:
                best_score = score
                best_moves = [(row, col)]
            elif score == best_score:
                best_moves.append((row, col))

        print(f"Branches explored: {self.branch_count}", flush=True)
        return random.choice(best_moves)

    def evaluate_position(self, player_bb: int, opponent_bb: int) -> float:
        """
        Evaluate the board based on:
        - mobility (number of valid moves),
        - corner control,
        - disc difference (in late game only).
        """
        total_discs = bin(player_bb | opponent_bb).count("1")

        def count_bits(bb: int) -> int:
            return bin(bb).count("1")

        # Game phase
        if total_discs <= 20:
            phase = 'early'
        elif total_discs <= 54:
            phase = 'mid'
        else:
            phase = 'late'

        # Mobility: normalized difference in move count
        my_moves = len(self.get_valid_moves_simple(player_bb, opponent_bb))
        opp_moves = len(self.get_valid_moves_simple(opponent_bb, player_bb))
        mobility = 0
        if my_moves + opp_moves > 0:
            mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

        # Corner control: stable positions
        corners = [0, 7, 56, 63]
        corner_mask = sum(1 << i for i in corners)
        corner_score = 25 * (count_bits(player_bb & corner_mask) - count_bits(opponent_bb & corner_mask))

        # Disc difference (only in late game)
        disc_score = 0
        if phase == 'late':
            player_discs = count_bits(player_bb)
            opponent_discs = count_bits(opponent_bb)
            disc_score = 100 * (player_discs - opponent_discs) / (player_discs + opponent_discs)

        if phase == 'early':
            return mobility + corner_score
        elif phase == 'mid':
            return mobility + 1.5 * corner_score
        else:
            return mobility + 2 * corner_score + disc_score