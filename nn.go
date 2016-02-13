package nn

import (
	"math"
	"math/rand"
)

type NN struct {
	Input   [][]float64
	Output  []float64
	Weights []float64
}

func Net(input [][]float64, output []float64) NN {
	return NN{input, output, Weights(len(input[0]))}
}

func Weights(length int) []float64 {
	rand.Seed(1)
	var ret = make([]float64, length)
	for i := 0; i < length; i++ {
		ret[i] = 2*rand.Float64() - 1
	}
	return ret
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidArray(x []float64) []float64 {
	return MapFloat64(x, Sigmoid)
}

func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func SigmoidPrimeArray(x []float64) []float64 {
	return MapFloat64(x, SigmoidPrime)
}

func Dot(x []float64, y []float64) float64 {
	r := 0.0
	if len(x) != len(y) {
		return math.Inf(-1)
	}
	for i, xi := range x {
		r += xi * y[i]
	}
	return r
}

func MatrixDot(x [][]float64, y [][]float64) []float64 {
	if len(x) != len(y) {
		return []float64{math.Inf(-1)}
	}
	r := make([]float64, len(x))
	for i, xi := range x {
		r[i] = Dot(xi, y[i])
	}
	return r
}

func PredictDot(x [][]float64, y []float64) []float64 {
	if len(x[0]) != len(y) {
		return []float64{math.Inf(-1)}
	}
	r := make([]float64, len(x))
	for i, xi := range x {
		r[i] = Dot(xi, y)
	}
	return r
}

func MapFloat64(vs []float64, f func(float64) float64) []float64 {
	vsm := make([]float64, len(vs))
	for i, v := range vs {
		vsm[i] = f(v)
	}
	return vsm
}

func Add(a []float64, b []float64) []float64 {
	if len(a) != len(b) {
		panic("Add not eqaul")
	}
	res := make([]float64, len(a))
	for i, ai := range a {
		res[i] = ai + b[i]
	}
	return res
}

func Subtract(a []float64, b []float64) []float64 {
	res := make([]float64, len(a))
	for i, ai := range a {
		res[i] = ai - b[i]
	}
	return res
}

func Multiply(a []float64, b []float64) []float64 {
	res := make([]float64, len(a))
	for i, ai := range a {
		res[i] = ai * b[i]
	}
	return res
}

func Transpose(a [][]float64) [][]float64 {
	b := make([][]float64, len(a[0]))
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[0]); j++ {
			b[j] = append(b[j], a[i][j])
		}
	}
	return b
}

func (n *NN) Train(iterations int) {
	for i := 0; i < iterations; i++ {
		l1 := SigmoidArray(PredictDot(n.Input, n.Weights))
		l1_error := Subtract(n.Output, l1)
		l1_delta := Multiply(l1_error, SigmoidPrimeArray(l1))
		n.Weights = Add(n.Weights, PredictDot(Transpose(n.Input), l1_delta))
	}
}

func (n NN) Predict(input [][]float64) []float64 {
	return SigmoidArray(PredictDot(input, n.Weights))
}
