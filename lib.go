package main

import "cmp"

type Integer interface {
	uint | uint8 | uint16 | uint32 | uint64 | int | int8 | int16 | int32 | int64
}

func MathMod[T Integer](n, mod T) T {
	if n < mod {
		return n
	} else {
		return n % mod
	}
}

func InRangeExclusive[T cmp.Ordered](n, minValue, maxValue T) bool {
	return n > minValue && n < maxValue
}

func InRangeInclusive[T cmp.Ordered](n, minValue, maxValue T) bool {
	return n >= minValue && n <= maxValue
}
