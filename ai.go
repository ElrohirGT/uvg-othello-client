package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
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
	BranchCount atomic.Uint64 // Track how many branches are evaluated during search
}

// NewSimpleOthelloAI creates a new AI instance
func NewSimpleOthelloAI() *SimpleOthelloAI {
	return &SimpleOthelloAI{BranchCount: atomic.Uint64{}}
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

// var validMovesCache = sync.Map{}

// GetValidMovesSimple iterates over all board positions to find valid moves
// for the current player based on the bitboard state.
func (ai *SimpleOthelloAI) GetValidMovesSimple(playerBB, opponentBB uint64) []Move {
	// validMovesCache.LoadOrStore(playerBB, &sync.Map{})
	// if innerCache, found := validMovesCache.Load(playerBB); found {
	// 	cache := innerCache.(*sync.Map)
	// 	if moves, found := cache.Load(opponentBB); found {
	// 		// log.Printf("CACHE HIT!")
	// 		return moves.([]Move)
	// 	}
	// }

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

	// validMovesCache[playerBB][opponentBB] = validMoves
	// cache, _ := validMovesCache.Load(playerBB)
	// cache.(*sync.Map).Store(opponentBB, validMoves)
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
		ai.BranchCount.Add(1)
		// ai.BranchCount++
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
	ai.BranchCount = atomic.Uint64{}
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

	parallelThreshold := 5
	shouldParalellize := len(validMoves) >= parallelThreshold
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

	if len(bestMoves) > 0 {
		// fmt.Printf("Branches explored: %d\n", ai.BranchCount)
		return bestMoves[rand.Intn(len(bestMoves))]
	} else {
		log.Print("Ran out of time for best moves! Returning a random valid move...")
		return validMoves[rand.Intn(len(validMoves))]
	}
}

var GAME_PHASE = struct {
	EARLY  int
	MIDDLE int
	LATE   int
}{
	EARLY:  0,
	MIDDLE: 1,
	LATE:   2,
}

// EvaluatePosition evaluates the board based on:
// - mobility (number of valid moves),
// - corner control,
// - disc difference (in late game only).
// EvaluatePosition evaluates the board with improved positional understanding
func (ai *SimpleOthelloAI) EvaluatePosition(playerBB, opponentBB uint64, playerValidMoveCount, oppValidMoveCount int) float64 {
	totalDiscs := bits.OnesCount64(playerBB | opponentBB)
	emptySquares := 64 - totalDiscs

	// Improved game phase detection
	phase := GAME_PHASE.LATE
	if totalDiscs <= 20 {
		phase = GAME_PHASE.EARLY
	} else if totalDiscs <= 50 { // Adjusted threshold
		phase = GAME_PHASE.MIDDLE
	}

	score := 0.0

	// 1. MOBILITY EVALUATION (Enhanced)
	mobilityScore := ai.evaluateMobility(playerBB, opponentBB, playerValidMoveCount, oppValidMoveCount, phase)
	
	// 2. POSITIONAL EVALUATION
	positionalScore := ai.evaluatePositional(playerBB, opponentBB, phase)
	
	// 3. STABILITY EVALUATION
	stabilityScore := ai.evaluateStability(playerBB, opponentBB)
	
	// 4. ENDGAME EVALUATION
	endgameScore := 0.0
	if phase == GAME_PHASE.LATE {
		endgameScore = ai.evaluateEndgame(playerBB, opponentBB, emptySquares)
	}

	// Phase-dependent weighting
	switch phase {
	case GAME_PHASE.EARLY:
		score = mobilityScore*1.0 + positionalScore*0.8 + stabilityScore*1.2
	case GAME_PHASE.MIDDLE:
		score = mobilityScore*0.8 + positionalScore*1.0 + stabilityScore*1.5
	default: // LATE
		score = mobilityScore*0.6 + positionalScore*0.8 + stabilityScore*2.0 + endgameScore*1.5
	}

	return score
}

// Enhanced mobility evaluation considering move quality
func (ai *SimpleOthelloAI) evaluateMobility(playerBB, opponentBB uint64, playerMoves, oppMoves int, phase int) float64 {
	if playerMoves+oppMoves == 0 {
		return 0
	}

	// Basic mobility
	basicMobility := 100.0 * float64(playerMoves-oppMoves) / float64(playerMoves+oppMoves)
	
	// Potential mobility (empty squares adjacent to opponent)
	playerPotential := ai.countPotentialMobility(playerBB, opponentBB)
	oppPotential := ai.countPotentialMobility(opponentBB, playerBB)
	
	potentialMobility := 0.0
	if playerPotential+oppPotential > 0 {
		potentialMobility = 50.0 * float64(playerPotential-oppPotential) / float64(playerPotential+oppPotential)
	}

	// Weight current vs potential mobility by phase
	if phase == GAME_PHASE.EARLY {
		return basicMobility*0.6 + potentialMobility*0.4
	}
	return basicMobility*0.8 + potentialMobility*0.2
}

// Count potential future mobility
func (ai *SimpleOthelloAI) countPotentialMobility(playerBB, opponentBB uint64) int {
	occupied := playerBB | opponentBB
	count := 0
	
	// Check empty squares adjacent to opponent pieces
	for pos := 0; pos < 64; pos++ {
		bit := uint64(1) << pos
		if occupied&bit != 0 {
			continue
		}
		
		// Check if adjacent to opponent
		if ai.isAdjacentTo(pos, opponentBB) {
			count++
		}
	}
	return count
}

// Enhanced positional evaluation with multiple factors
func (ai *SimpleOthelloAI) evaluatePositional(playerBB, opponentBB uint64, phase int) float64 {
	score := 0.0
	
	// 1. Corner control (most important)
	cornerScore := ai.evaluateCorners(playerBB, opponentBB)
	
	// 2. Edge control
	edgeScore := ai.evaluateEdges(playerBB, opponentBB)
	
	// 3. X-squares and C-squares (dangerous early, good late)
	xCSquareScore := ai.evaluateXCSquares(playerBB, opponentBB, phase)
	
	// 4. Center control (important early)
	centerScore := ai.evaluateCenter(playerBB, opponentBB, phase)

	score = cornerScore*4.0 + edgeScore*2.0 + xCSquareScore + centerScore
	return score
}

// Corner evaluation with adjacent square consideration
func (ai *SimpleOthelloAI) evaluateCorners(playerBB, opponentBB uint64) float64 {
	corners := []int{0, 7, 56, 63}
	var cornerMask uint64
	for _, pos := range corners {
		cornerMask |= uint64(1) << pos
	}
	
	playerCorners := bits.OnesCount64(playerBB & cornerMask)
	oppCorners := bits.OnesCount64(opponentBB & cornerMask)
	
	return 25.0 * float64(playerCorners - oppCorners)
}

// Edge evaluation (excluding corners)
func (ai *SimpleOthelloAI) evaluateEdges(playerBB, opponentBB uint64) float64 {
	edgeMask := uint64(0xFF818181818181FF) // All edge squares
	cornerMask := uint64(0x8100000000000081) // Corner squares
	pureEdgeMask := edgeMask &^ cornerMask // Edges without corners
	
	playerEdges := bits.OnesCount64(playerBB & pureEdgeMask)
	oppEdges := bits.OnesCount64(opponentBB & pureEdgeMask)
	
	return 5.0 * float64(playerEdges - oppEdges)
}

// X-squares and C-squares evaluation
func (ai *SimpleOthelloAI) evaluateXCSquares(playerBB, opponentBB uint64, phase int) float64 {
	// X-squares (diagonally adjacent to corners)
	xSquareMask := uint64(0x0042000000004200)
	// C-squares (orthogonally adjacent to corners)  
	cSquareMask := uint64(0x0081424242810000)
	
	playerX := bits.OnesCount64(playerBB & xSquareMask)
	oppX := bits.OnesCount64(opponentBB & xSquareMask)
	playerC := bits.OnesCount64(playerBB & cSquareMask)
	oppC := bits.OnesCount64(opponentBB & cSquareMask)
	
	xScore := float64(playerX - oppX)
	cScore := float64(playerC - oppC)
	
	// These squares are bad early, neutral late
	multiplier := -3.0 // Bad early
	if phase == GAME_PHASE.LATE {
		multiplier = 0.5 // Slightly good late
	} else if phase == GAME_PHASE.MIDDLE {
		multiplier = -1.0 // Less bad in middle
	}
	
	return multiplier * (xScore*2.0 + cScore*1.0)
}

// Center control evaluation
func (ai *SimpleOthelloAI) evaluateCenter(playerBB, opponentBB uint64, phase int) float64 {
	// Central 4x4 area
	centerMask := uint64(0x0000001818000000)
	
	playerCenter := bits.OnesCount64(playerBB & centerMask)
	oppCenter := bits.OnesCount64(opponentBB & centerMask)
	
	multiplier := 2.0 // Good early
	if phase == GAME_PHASE.MIDDLE {
		multiplier = 1.0
	} else if phase == GAME_PHASE.LATE {
		multiplier = 0.5 // Less important late
	}
	
	return multiplier * float64(playerCenter - oppCenter)
}

// Basic stability evaluation
func (ai *SimpleOthelloAI) evaluateStability(playerBB, opponentBB uint64) float64 {
	playerStable := ai.countStableDiscs(playerBB, opponentBB)
	oppStable := ai.countStableDiscs(opponentBB, playerBB)
	
	return 10.0 * float64(playerStable - oppStable)
}

// Count stable discs (simplified - just corners and secure edges for now)
func (ai *SimpleOthelloAI) countStableDiscs(playerBB, opponentBB uint64) int {
	stable := uint64(0)
	
	// Corners are always stable if occupied
	corners := []int{0, 7, 56, 63}
	for _, corner := range corners {
		bit := uint64(1) << corner
		if playerBB&bit != 0 {
			stable |= bit
		}
	}
	
	// Add simple edge stability (connected to corners)
	stable |= ai.findStableEdges(playerBB, opponentBB, stable)
	
	return bits.OnesCount64(stable)
}

// Find stable edges connected to stable corners
func (ai *SimpleOthelloAI) findStableEdges(playerBB, opponentBB uint64, stable uint64) uint64 {
	newStable := stable
	occupied := playerBB | opponentBB
	
	// Check each edge from corners
	edges := [][]int{
		{0, 1, 2, 3, 4, 5, 6, 7},     // Top edge
		{56, 57, 58, 59, 60, 61, 62, 63}, // Bottom edge
		{0, 8, 16, 24, 32, 40, 48, 56},   // Left edge
		{7, 15, 23, 31, 39, 47, 55, 63},  // Right edge
	}
	
	for _, edge := range edges {
		ai.addStableEdgeDiscs(&newStable, playerBB, occupied, edge)
	}
	
	return newStable
}

// Helper to add stable discs along an edge
func (ai *SimpleOthelloAI) addStableEdgeDiscs(stable *uint64, playerBB, occupied uint64, edge []int) {
	// From left
	for i := 0; i < len(edge); i++ {
		bit := uint64(1) << edge[i]
		if playerBB&bit != 0 && (*stable&bit != 0 || i == 0) {
			*stable |= bit
		} else if occupied&bit == 0 {
			break
		}
	}
	
	// From right
	for i := len(edge) - 1; i >= 0; i-- {
		bit := uint64(1) << edge[i]
		if playerBB&bit != 0 && (*stable&bit != 0 || i == len(edge)-1) {
			*stable |= bit
		} else if occupied&bit == 0 {
			break
		}
	}
}

// Endgame evaluation with parity and exact counting
func (ai *SimpleOthelloAI) evaluateEndgame(playerBB, opponentBB uint64, emptySquares int) float64 {
	playerDiscs := bits.OnesCount64(playerBB)
	opponentDiscs := bits.OnesCount64(opponentBB)
	
	// Basic disc difference
	discDiff := 100.0 * float64(playerDiscs-opponentDiscs) / float64(playerDiscs+opponentDiscs)
	
	// Parity consideration (who gets the last move)
	parityBonus := 0.0
	if emptySquares <= 10 && emptySquares%2 == 1 {
		parityBonus = 5.0 // Slight bonus for having last move
	}
	
	return discDiff + parityBonus
}

// Helper function to check if position is adjacent to any piece in bitboard
func (ai *SimpleOthelloAI) isAdjacentTo(pos int, bitboard uint64) bool {
	row, col := pos/8, pos%8
	
	for dr := -1; dr <= 1; dr++ {
		for dc := -1; dc <= 1; dc++ {
			if dr == 0 && dc == 0 {
				continue
			}
			
			newRow, newCol := row+dr, col+dc
			if newRow >= 0 && newRow < 8 && newCol >= 0 && newCol < 8 {
				adjPos := newRow*8 + newCol
				if bitboard&(uint64(1)<<adjPos) != 0 {
					return true
				}
			}
		}
	}
	return false
}

var timeout = time.Second * 3

// Example usage
func main() {

	// var playerBB uint64
	// flag.Uint64Var(&playerBB, "pBB", 0, "The player bitboard")

	// var opponentBB uint64
	// flag.Uint64Var(&opponentBB, "opBB", 0, "The opponent bitboard")

	var searchDepth int
	flag.IntVar(&searchDepth, "s", SEARCH_DEPTH, "The search depth")

	var cpuprofile string
	flag.StringVar(&cpuprofile, "cpuprofile", "", "write cpu profile to file")

	var memprofile string
	flag.StringVar(&memprofile, "memprofile", "", "write memory profile to file")
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

	reader := bufio.NewReader(os.Stdin)
	for {
		context, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatalf("An error occurred reading line from stdin: %s", err)
		}

		parts := strings.Split(string(line), " ")

		playerBB, err := strconv.ParseUint(parts[0], 10, 64)
		if err != nil {
			log.Fatalf("An error occurred parsing playerBB: %s", err)
		}

		opponentBB, err := strconv.ParseUint(parts[1], 10, 64)
		if err != nil {
			log.Fatalf("An error occurred parsing opponentBB: %s", err)
		}

		bestMove := ai.SelectBestMove(context, uint64(playerBB), uint64(opponentBB), searchDepth)
		log.Printf("Analized %d branches", ai.BranchCount.Load())
		fmt.Printf("%d %d\n", bestMove.Row, bestMove.Col)
		cancel()
	}

	if memprofile != "" {
		f, err := os.Create(memprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.WriteHeapProfile(f)
		f.Close()
	}
}
