package main

import (
	"fmt"
	"log"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"strconv"
)

// Configuration constants
const SEARCH_DEPTH = 6 // Default search depth for minimax

// Move represents a board position
type Move struct {
	Row, Col int
}

// SimpleOthelloAI represents the AI player
type SimpleOthelloAI struct {
	BranchCount int // Track how many branches are evaluated during search
}

// NewSimpleOthelloAI creates a new AI instance
func NewSimpleOthelloAI() *SimpleOthelloAI {
	return &SimpleOthelloAI{BranchCount: 0}
}

// BoardToBitboard converts a 2D board into two 64-bit integers (bitboards),
// one for the current player and one for the opponent.
func (ai *SimpleOthelloAI) BoardToBitboard(board [][]int, player int) (uint64, uint64) {
	var playerBB, opponentBB uint64
	opponent := -player

	for row := range 8 {
		for col := range 8 {
			pos := uint64(1) << (row*8 + col)
			if board[row][col] == player {
				playerBB |= pos
			} else if board[row][col] == opponent {
				opponentBB |= pos
			}
		}
	}

	return playerBB, opponentBB
}

// BitboardToBoard converts bitboards back to a 2D board representation.
// Used for visualization or fallback.
func (ai *SimpleOthelloAI) BitboardToBoard(playerBB, opponentBB uint64, player int) [][]int {
	board := make([][]int, 8)
	for i := range board {
		board[i] = make([]int, 8)
	}

	opponent := -player

	for row := range 8 {
		for col := range 8 {
			pos := uint64(1) << (row*8 + col)
			if playerBB&pos != 0 {
				board[row][col] = player
			} else if opponentBB&pos != 0 {
				board[row][col] = opponent
			}
		}
	}

	return board
}

// GetValidMovesSimple iterates over all board positions to find valid moves
// for the current player based on the bitboard state.
func (ai *SimpleOthelloAI) GetValidMovesSimple(playerBB, opponentBB uint64) []Move {
	var validMoves []Move
	occupied := playerBB | opponentBB

	for pos := range 64 {
		bit := uint64(1) << pos
		if occupied&bit != 0 {
			continue
		}

		if ai.IsValidMoveBitboard(pos, playerBB, opponentBB) {
			row, col := pos/8, pos%8
			validMoves = append(validMoves, Move{Row: row, Col: col})
		}
	}

	return validMoves
}

// IsValidMoveBitboard checks whether placing a piece at a given position is a valid move.
// Looks in all 8 directions to find if at least one opponent piece is flanked.
func (ai *SimpleOthelloAI) IsValidMoveBitboard(movePos int, playerBB, opponentBB uint64) bool {
	if (playerBB|opponentBB)&(uint64(1)<<movePos) != 0 {
		return false // Already occupied
	}

	directions := []int{1, -1, 8, -8, 9, -9, 7, -7} // 8 surrounding directions

	for _, direction := range directions {
		pos := movePos
		foundOpponent := false

		for {
			// Break if we cross board boundaries
			if direction == -1 && pos%8 == 0 {
				break
			}
			if direction == 1 && pos%8 == 7 {
				break
			}
			if direction == -9 && (pos%8 == 0 || pos < 8) {
				break
			}
			if direction == -8 && pos < 8 {
				break
			}
			if direction == -7 && (pos%8 == 7 || pos < 8) {
				break
			}
			if direction == 7 && (pos%8 == 0 || pos > 55) {
				break
			}
			if direction == 8 && pos > 55 {
				break
			}
			if direction == 9 && (pos%8 == 7 || pos > 55) {
				break
			}

			pos += direction
			if pos < 0 || pos > 63 {
				break
			}

			bit := uint64(1) << pos
			if opponentBB&bit != 0 {
				foundOpponent = true
				continue
			} else if playerBB&bit != 0 {
				if foundOpponent {
					return true // Found a bracketed line
				}
				break
			} else {
				break
			}
		}
	}

	return false // No direction formed a valid capture
}

// BitboardToMove converts a bitboard with a single set bit into (row, col) coordinates.
func (ai *SimpleOthelloAI) BitboardToMove(moveBitboard uint64) *Move {
	if moveBitboard == 0 {
		return nil
	}

	pos := bits.TrailingZeros64(moveBitboard)
	return &Move{Row: pos / 8, Col: pos % 8}
}

// PrintBitboard prints the bitboard in a human-readable 8x8 format.
// Used for debugging.
func (ai *SimpleOthelloAI) PrintBitboard(bitboard uint64, label string) {
	if label != "" {
		fmt.Printf("%s:\n", label)
	}

	for row := range 8 {
		rowStr := ""
		for col := range 8 {
			pos := uint64(1) << (row*8 + col)
			if bitboard&pos != 0 {
				rowStr += "1 "
			} else {
				rowStr += "0 "
			}
		}
		fmt.Println(rowStr)
	}
	fmt.Println()
}

// ApplyMove applies a move at movePos to flip appropriate discs.
// Returns updated bitboards after move execution.
func (ai *SimpleOthelloAI) ApplyMove(movePos int, playerBB, opponentBB uint64) (uint64, uint64) {
	var flipped uint64
	directions := []int{1, -1, 8, -8, 9, -9, 7, -7}

	for _, direction := range directions {
		pos := movePos
		var captured uint64

		for {
			// Boundary checks
			if direction == -1 && pos%8 == 0 {
				break
			}
			if direction == 1 && pos%8 == 7 {
				break
			}
			if direction == -9 && (pos%8 == 0 || pos < 8) {
				break
			}
			if direction == -8 && pos < 8 {
				break
			}
			if direction == -7 && (pos%8 == 7 || pos < 8) {
				break
			}
			if direction == 7 && (pos%8 == 0 || pos > 55) {
				break
			}
			if direction == 8 && pos > 55 {
				break
			}
			if direction == 9 && (pos%8 == 7 || pos > 55) {
				break
			}

			pos += direction
			if pos < 0 || pos > 63 {
				break
			}

			bit := uint64(1) << pos
			if opponentBB&bit != 0 {
				captured |= bit
			} else if playerBB&bit != 0 {
				flipped |= captured
				break
			} else {
				break
			}
		}
	}

	moveBit := uint64(1) << movePos
	return playerBB | moveBit | flipped, opponentBB &^ flipped
}

// Minimax implements basic minimax without pruning.
// Uses negamax convention (scores are negated across recursion).
func (ai *SimpleOthelloAI) Minimax(playerBB, opponentBB uint64, depth int, maximizingPlayer bool) float64 {
	if depth == 0 {
		return ai.EvaluatePosition(playerBB, opponentBB)
	}

	validMoves := ai.GetValidMovesSimple(playerBB, opponentBB)

	if len(validMoves) == 0 {
		if len(ai.GetValidMovesSimple(opponentBB, playerBB)) > 0 {
			return -ai.Minimax(opponentBB, playerBB, depth, !maximizingPlayer)
		} else {
			return ai.EvaluatePosition(playerBB, opponentBB)
		}
	}

	bestScore := math.Inf(-1)
	for _, move := range validMoves {
		ai.BranchCount++
		movePos := move.Row*8 + move.Col
		newPlayerBB, newOpponentBB := ai.ApplyMove(movePos, playerBB, opponentBB)
		score := -ai.Minimax(newOpponentBB, newPlayerBB, depth-1, !maximizingPlayer)
		bestScore = math.Max(bestScore, score)
	}

	return bestScore
}

// MinimaxAlphaBeta implements minimax with alpha-beta pruning.
// Prunes branches that can't improve the outcome.
func (ai *SimpleOthelloAI) MinimaxAlphaBeta(playerBB, opponentBB uint64, depth int, alpha, beta float64, maximizingPlayer bool) float64 {
	if depth == 0 {
		return ai.EvaluatePosition(playerBB, opponentBB)
	}

	validMoves := ai.GetValidMovesSimple(playerBB, opponentBB)

	if len(validMoves) == 0 {
		if len(ai.GetValidMovesSimple(opponentBB, playerBB)) > 0 {
			return -ai.MinimaxAlphaBeta(opponentBB, playerBB, depth, -beta, -alpha, !maximizingPlayer)
		} else {
			return ai.EvaluatePosition(playerBB, opponentBB)
		}
	}

	bestScore := math.Inf(-1)
	for _, move := range validMoves {
		ai.BranchCount++
		movePos := move.Row*8 + move.Col
		newPlayerBB, newOpponentBB := ai.ApplyMove(movePos, playerBB, opponentBB)
		score := -ai.MinimaxAlphaBeta(newOpponentBB, newPlayerBB, depth-1, -beta, -alpha, !maximizingPlayer)
		bestScore = math.Max(bestScore, score)
		alpha = math.Max(alpha, bestScore)
		if alpha >= beta {
			break // Beta cutoff
		}
	}

	return bestScore
}

// SelectBestMove chooses the best move by searching game tree using minimax with alpha-beta pruning.
// Randomly breaks ties between equally good moves.
func (ai *SimpleOthelloAI) SelectBestMove(playerBB, opponentBB uint64, depth int) Move {
	ai.BranchCount = 0
	validMoves := ai.GetValidMovesSimple(playerBB, opponentBB)

	if len(validMoves) == 0 {
		// Fallback: pick random empty square if no valid moves (unusual case)
		var emptySquares []Move
		for r := range 8 {
			for c := range 8 {
				if (playerBB|opponentBB)&(uint64(1)<<(r*8+c)) == 0 {
					emptySquares = append(emptySquares, Move{Row: r, Col: c})
				}
			}
		}
		if len(emptySquares) > 0 {
			return emptySquares[rand.Intn(len(emptySquares))]
		}
		return Move{Row: 0, Col: 0}
	}

	bestScore := math.Inf(-1)
	var bestMoves []Move

	for _, move := range validMoves {
		movePos := move.Row*8 + move.Col
		newPlayerBB, newOpponentBB := ai.ApplyMove(movePos, playerBB, opponentBB)
		score := -ai.MinimaxAlphaBeta(newOpponentBB, newPlayerBB, depth-1, math.Inf(-1), math.Inf(1), false)
		// Uncomment to use normal minimax function without pruning
		// score := -ai.Minimax(newOpponentBB, newPlayerBB, depth-1, false)

		if score > bestScore {
			bestScore = score
			bestMoves = []Move{move}
		} else if score == bestScore {
			bestMoves = append(bestMoves, move)
		}
	}

	// fmt.Printf("Branches explored: %d\n", ai.BranchCount)
	return bestMoves[rand.Intn(len(bestMoves))]
}

// EvaluatePosition evaluates the board based on:
// - mobility (number of valid moves),
// - corner control,
// - disc difference (in late game only).
func (ai *SimpleOthelloAI) EvaluatePosition(playerBB, opponentBB uint64) float64 {
	totalDiscs := bits.OnesCount64(playerBB | opponentBB)

	// Game phase
	var phase string
	if totalDiscs <= 20 {
		phase = "early"
	} else if totalDiscs <= 54 {
		phase = "mid"
	} else {
		phase = "late"
	}

	// Mobility: normalized difference in move count
	myMoves := len(ai.GetValidMovesSimple(playerBB, opponentBB))
	oppMoves := len(ai.GetValidMovesSimple(opponentBB, playerBB))
	mobility := 0.0
	if myMoves+oppMoves > 0 {
		mobility = 100.0 * float64(myMoves-oppMoves) / float64(myMoves+oppMoves)
	}

	// Corner control: stable positions
	corners := []int{0, 7, 56, 63}
	var cornerMask uint64
	for _, i := range corners {
		cornerMask |= uint64(1) << i
	}
	cornerScore := 25.0 * float64(bits.OnesCount64(playerBB&cornerMask)-bits.OnesCount64(opponentBB&cornerMask))

	// Disc difference (only in late game)
	discScore := 0.0
	if phase == "late" {
		playerDiscs := bits.OnesCount64(playerBB)
		opponentDiscs := bits.OnesCount64(opponentBB)
		discScore = 100.0 * float64(playerDiscs-opponentDiscs) / float64(playerDiscs+opponentDiscs)
	}

	switch phase {
	case "early":
		return mobility + cornerScore
	case "mid":
		return mobility + 1.5*cornerScore
	default: // late
		return mobility + 2*cornerScore + discScore
	}
}

// Example usage
func main() {
	// rand.Seed(time.Now().UnixNano())

	ai := NewSimpleOthelloAI()

	// Example board setup (standard Othello starting position)
	// board := [][]int{
	// 	{0, 0, 0, 0, 0, 0, 0, 0},
	// 	{0, 0, 0, 0, 0, 0, 0, 0},
	// 	{0, 0, 0, 0, 0, 0, 0, 0},
	// 	{0, 0, 0, -1, 1, 0, 0, 0},
	// 	{0, 0, 0, 1, -1, 0, 0, 0},
	// 	{0, 0, 0, 0, 0, 0, 0, 0},
	// 	{0, 0, 0, 0, 0, 0, 0, 0},
	// 	{0, 0, 0, 0, 0, 0, 0, 0},
	// }

	// player := 1
	// playerBB, opponentBB := ai.BoardToBitboard(board, player)

	// log.Printf("Trying to parse: %s", os.Args[1])
	playerBB, err := strconv.ParseUint(os.Args[1], 10, 64)
	if err != nil {
		log.Fatalf("Invalid playerBB received! %s", err)
	}
	opponentBB, err := strconv.ParseUint(os.Args[2], 10, 64)
	if err != nil {
		log.Fatalf("Invalid opponentBB received! %s", err)
	}
	searchDepth, err := strconv.Atoi(os.Args[3])
	if err != nil {
		log.Fatalf("Invalid searchDepth received! %s", err)
	}

	bestMove := ai.SelectBestMove(uint64(playerBB), uint64(opponentBB), searchDepth)
	fmt.Printf("%d %d", bestMove.Row, bestMove.Col)
}
