package main

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
)

func BenchmarkSelectBestMove(b *testing.B) {
	filePath := "./inputs/sequenceOfMoves"
	fileBytes, err := os.ReadFile(filePath)
	if err != nil {
		b.Fatalf("Failed to read file %s", filePath)
	}

	playerBBs := []uint64{}
	opponentBBs := []uint64{}
	for line := range strings.SplitSeq(string(fileBytes), "\n") {
		parts := strings.Split(line, " ")
		if parts[0] == "" {
			continue
		}

		playerBB, err := strconv.ParseUint(parts[0], 10, 64)
		if err != nil {
			b.Fatalf("An error occurred parsing playerBB: %s", err)
		}
		playerBBs = append(playerBBs, playerBB)

		opponentBB, err := strconv.ParseUint(parts[1], 10, 64)
		if err != nil {
			b.Fatalf("An error occurred parsing opponentBB: %s", err)
		}
		opponentBBs = append(opponentBBs, opponentBB)
	}

	ai := NewSimpleOthelloAI()

	for b.Loop() {
		for i := range len(opponentBBs) {
			ai.SelectBestMove(context.Background(), playerBBs[i], opponentBBs[i], 7)
		}
	}

}
