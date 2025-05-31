package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"runtime/pprof"
	"strings"
	"sync"
	"time"
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
func BoardToBitboard(board [][]int, player int) (uint64, uint64) {
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
func BitboardToBoard(playerBB, opponentBB uint64, player int) [][]int {
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

var centerMask, _ = BoardToBitboard([][]int{
	{1, 1, 1, 0, 0, 0, 0, 0},
	{1, 1, 1, 0, 0, 0, 0, 0},
	{1, 1, 1, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
}, 1)
var topBottomMask, _ = BoardToBitboard([][]int{
	{1, 1, 1, 0, 0, 0, 0, 0},
	{1, 1, 1, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
}, 1)
var sidesMask, _ = BoardToBitboard([][]int{
	{1, 1, 0, 0, 0, 0, 0, 0},
	{1, 1, 0, 0, 0, 0, 0, 0},
	{1, 1, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
}, 1)
var cornerMask, _ = BoardToBitboard([][]int{
	{1, 1, 0, 0, 0, 0, 0, 0},
	{1, 1, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
}, 1)

func GetNeighboursFor(pos uint64) uint64 {

	bitIsSide := MathMod(pos, 8) == 0 || MathMod(pos, 8) == 7
	bitIsTopBottom := InRangeInclusive(pos, 0, 7) || InRangeInclusive(pos, 8*7, 8*8-1)

	mask := centerMask
	maskCenter := uint64(8 + 1)
	if pos == 0 {
		mask = cornerMask
		maskCenter = 0
	} else if pos == 7 {
		mask = cornerMask << 6
		maskCenter = 7
	} else if pos == 8*7 {
		mask = cornerMask << (7*7 - 1)
		maskCenter = 8 * 7
	} else if pos == 63 {
		mask = cornerMask << (7*7 + 5)
		maskCenter = 63
	} else if bitIsSide {
		mask = sidesMask
		maskCenter = 8
		if MathMod(pos, 8) == 7 {
			maskCenter = 9
		}
	} else if bitIsTopBottom {
		mask = topBottomMask
		maskCenter = 1
		if InRangeInclusive(pos, 8*7, 8*8-1) {
			maskCenter = 9
		}
	}

	delta := pos - maskCenter
	if delta < 0 {
		mask = mask >> delta
	} else {
		mask = mask << delta
	}

	// log.Print(PrintBitboard(mask, fmt.Sprintf("Idx: %d, Delta: %d", pos, delta)))

	return mask
}

func GetNeighboursBitboards(opponentBB uint64) []uint64 {
	bitboards := make([]uint64, 8*8/2)
	for pos := range uint64(64) {
		bit := uint64(1) << pos
		if opponentBB&bit == 0 {
			continue
		}

		bitboard := GetNeighboursFor(pos)
		bitboards = append(bitboards, bitboard)
	}
	return bitboards
}

// GetValidMovesSimple iterates over all board positions to find valid moves
// for the current player based on the bitboard state.
func (ai *SimpleOthelloAI) GetValidMovesSimple(playerBB, opponentBB uint64) []Move {
	validMoves := make([]Move, 0, 10)
	occupied := playerBB | opponentBB
	neighboursBitboards := GetNeighboursBitboards(opponentBB)

	for pos := range 64 {
		bit := uint64(1) << pos
		if occupied&bit != 0 {
			continue
		}

		isNeighbourToOpponentPiece := false
		for _, neighbourBB := range neighboursBitboards {
			foundInBitboard := neighbourBB&bit != 0
			if foundInBitboard {
				isNeighbourToOpponentPiece = true
				break
			}
		}
		if !isNeighbourToOpponentPiece {
			// log.Printf("Pos: %d is NOT a neighbour of an opponent piece!", pos)
			continue
		}

		// log.Printf("Pos: %d is a neighbour of an opponent piece!", pos)

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
func PrintBitboard(bitboard uint64, label string) string {
	b := strings.Builder{}
	if label != "" {
		b.WriteString(label)
		b.WriteString(":\n")
	}

	for row := range 8 {
		for col := range 8 {
			pos := uint64(1) << (row*8 + col)
			if bitboard&pos != 0 {
				b.WriteString("1 ")
			} else {
				b.WriteString("0 ")
			}
		}
		b.WriteRune('\n')
	}
	b.WriteRune('\n')

	return b.String()
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
// func (ai *SimpleOthelloAI) Minimax(playerBB, opponentBB uint64, depth int, maximizingPlayer bool) float64 {
// 	if depth == 0 {
// 		return ai.EvaluatePosition(playerBB, opponentBB)
// 	}
//
// 	validMoves := ai.GetValidMovesSimple(playerBB, opponentBB)
//
// 	if len(validMoves) == 0 {
// 		if len(ai.GetValidMovesSimple(opponentBB, playerBB)) > 0 {
// 			return -ai.Minimax(opponentBB, playerBB, depth, !maximizingPlayer)
// 		} else {
// 			return ai.EvaluatePosition(playerBB, opponentBB)
// 		}
// 	}
//
// 	bestScore := math.Inf(-1)
// 	for _, move := range validMoves {
// 		ai.BranchCount++
// 		movePos := move.Row*8 + move.Col
// 		newPlayerBB, newOpponentBB := ai.ApplyMove(movePos, playerBB, opponentBB)
// 		score := -ai.Minimax(newOpponentBB, newPlayerBB, depth-1, !maximizingPlayer)
// 		bestScore = math.Max(bestScore, score)
// 	}
//
// 	return bestScore
// }

// MinimaxAlphaBeta implements minimax with alpha-beta pruning.
// Prunes branches that can't improve the outcome.
func (ai *SimpleOthelloAI) MinimaxAlphaBeta(context context.Context, playerBB, opponentBB uint64, depth int, alpha, beta float64, maximizingPlayer bool) float64 {
	if context.Err() != nil {
		return 0
	}

	playerValidMoves := ai.GetValidMovesSimple(playerBB, opponentBB)
	oppValidMoves := ai.GetValidMovesSimple(opponentBB, playerBB)

	if depth == 0 {
		return ai.EvaluatePosition(playerBB, opponentBB, len(playerValidMoves), len(oppValidMoves))
	}

	if len(playerValidMoves) == 0 {
		if len(oppValidMoves) > 0 {
			return -ai.MinimaxAlphaBeta(context, opponentBB, playerBB, depth, -beta, -alpha, !maximizingPlayer)
		} else {
			return ai.EvaluatePosition(playerBB, opponentBB, len(playerValidMoves), len(oppValidMoves))
		}
	}

	bestScore := math.Inf(-1)
	for _, move := range playerValidMoves {
		ai.BranchCount++
		movePos := move.Row*8 + move.Col
		newPlayerBB, newOpponentBB := ai.ApplyMove(movePos, playerBB, opponentBB)
		score := -ai.MinimaxAlphaBeta(context, newOpponentBB, newPlayerBB, depth-1, -beta, -alpha, !maximizingPlayer)
		if context.Err() != nil {
			break
		}

		bestScore = math.Max(bestScore, score)
		alpha = math.Max(alpha, bestScore)
		if alpha >= beta {
			break // Beta cutoff
		}
	}

	return bestScore
}

type MiniMaxResult struct {
	Score float64
	Move  Move
}

// SelectBestMove chooses the best move by searching game tree using minimax with alpha-beta pruning.
// Randomly breaks ties between equally good moves.
func (ai *SimpleOthelloAI) SelectBestMove(context context.Context, playerBB, opponentBB uint64, depth int) Move {
	ai.BranchCount = 0
	validMoves := ai.GetValidMovesSimple(playerBB, opponentBB)

	if len(validMoves) == 0 {
		log.Print("Selecting a random movement...")
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
	bestMoves := make([]Move, 0, len(validMoves))

	syncLimitMoveCount := 4
	shouldParalellize := len(validMoves) > syncLimitMoveCount
	if shouldParalellize {
		// log.Printf("Too many valid moves (>%d), using go routines...", syncLimitMoveCount)
		minMaxResultChan := make(chan *MiniMaxResult, len(validMoves))

		cumGroup := sync.WaitGroup{}

		// CUMulative thread
		cumGroup.Add(1)
		go func() {
			defer cumGroup.Done()

			for {
				select {
				case <-context.Done():
					return
				case result := <-minMaxResultChan:
					if result == nil {
						return
					}

					if result.Score > bestScore {
						bestScore = result.Score
						bestMoves = []Move{result.Move}
					} else if result.Score == bestScore {
						bestMoves = append(bestMoves, result.Move)
					}
				}
			}
		}()

		workerGroup := sync.WaitGroup{}
		// High level movement threads
		for _, move := range validMoves {
			workerGroup.Add(1)
			go func() {
				defer func() {
					// log.Printf("Thread for move %d finished!", i)
					workerGroup.Done()
				}()

				movePos := move.Row*8 + move.Col
				newPlayerBB, newOpponentBB := ai.ApplyMove(movePos, playerBB, opponentBB)
				score := -ai.MinimaxAlphaBeta(context, newOpponentBB, newPlayerBB, depth-1, math.Inf(-1), math.Inf(1), false)
				if context.Err() != nil {
					return
				}
				// Uncomment to use normal minimax function without pruning
				// score := -ai.Minimax(newOpponentBB, newPlayerBB, depth-1, false)

				minMaxResultChan <- &MiniMaxResult{
					Score: score,
					Move:  move,
				}
			}()
		}

		// log.Print("Waiting for threads to finish...")
		workerGroup.Wait()
		close(minMaxResultChan)
		cumGroup.Wait()
	} else {
		for _, move := range validMoves {
			movePos := move.Row*8 + move.Col
			newPlayerBB, newOpponentBB := ai.ApplyMove(movePos, playerBB, opponentBB)
			score := -ai.MinimaxAlphaBeta(context, newOpponentBB, newPlayerBB, depth-1, math.Inf(-1), math.Inf(1), false)
			if context.Err() != nil {
				break
			}
			// Uncomment to use normal minimax function without pruning
			// score := -ai.Minimax(newOpponentBB, newPlayerBB, depth-1, false)

			if score > bestScore {
				bestScore = score
				bestMoves = []Move{move}
			} else if score == bestScore {
				bestMoves = append(bestMoves, move)
			}
		}
	}
	if context.Err() != nil {
		log.Print("Reached timeout of ", timeout)
	}

	// fmt.Printf("Branches explored: %d\n", ai.BranchCount)
	return bestMoves[rand.Intn(len(bestMoves))]
}

// EvaluatePosition evaluates the board based on:
// - mobility (number of valid moves),
// - corner control,
// - disc difference (in late game only).
func (ai *SimpleOthelloAI) EvaluatePosition(playerBB, opponentBB uint64, playerValidMoveCount, oppValidMoveCount int) float64 {
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
	// myMoves := len(ai.GetValidMovesSimple(playerBB, opponentBB))
	// oppMoves := len(ai.GetValidMovesSimple(opponentBB, playerBB))
	mobility := 0.0
	if playerValidMoveCount+oppValidMoveCount > 0 {
		mobility = 100.0 * float64(playerValidMoveCount-oppValidMoveCount) / float64(playerValidMoveCount+oppValidMoveCount)
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

var timeout = time.Second * 3

// Example usage
func main() {
	context, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	var playerBB uint64
	flag.Uint64Var(&playerBB, "pBB", 0, "The player bitboard")

	var opponentBB uint64
	flag.Uint64Var(&opponentBB, "opBB", 0, "The opponent bitboard")

	var searchDepth int
	flag.IntVar(&searchDepth, "s", SEARCH_DEPTH, "The search depth")

	var cpuprofile string
	flag.StringVar(&cpuprofile, "cpuprofile", "", "write cpu profile to file")
	flag.Parse()

	if cpuprofile != "" {
		log.Print("Profiling program in: ", cpuprofile)
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

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

	bestMove := ai.SelectBestMove(context, uint64(playerBB), uint64(opponentBB), searchDepth)
	fmt.Printf("%d %d", bestMove.Row, bestMove.Col)
}
