package nn_test

import (
	"fmt"
	"github.com/r0fls/nngo"
)

func ExampleNN() {
	input := [][]float64{[]float64{0, 0, 1}, []float64{0, 1, 1}, []float64{1, 0, 1}, []float64{1, 1, 1}}
	j := nn.Net(input, []float64{0, 0, 1, 1})
	fmt.Println(j.Output)
	// Output: [0 0 1 1]
}

func ExamplePredict() {
	input := [][]float64{[]float64{0, 0, 1}, []float64{0, 1, 1}, []float64{1, 0, 1}, []float64{1, 1, 1}}
	j := nn.Net(input, []float64{0, 0, 1, 1})
	j.Train(10000)
	fmt.Println(j.Predict(input))
	// Output: [0.0007218168710223273 0.0004809626072683371 0.9995924639241274 0.999388357363272]
}
func ExampleTranspose() {
	input := [][]float64{[]float64{0, 0, 1}, []float64{0, 1, 1}, []float64{1, 0, 1}, []float64{1, 1, 1}}
	fmt.Println(nn.Transpose(input))
	// Output: [[0 0 1 1] [0 1 0 1] [1 1 1 1]]
}

func ExampleMult() {
	j := []float64{0, 0, 1, 1}
	fmt.Println(nn.Multiply(j, j))
	// Output: [0 0 1 1]
}

func ExampleSubtract() {
	j := []float64{0, 0, 1, 1}
	fmt.Println(nn.Subtract(j, j))
	// Output: [0 0 0 0]
}

func ExamplePredictDot() {
	input := [][]float64{[]float64{0, 0, 1}, []float64{0, 1, 1}, []float64{1, 0, 1}, []float64{1, 1, 1}}
	j := []float64{0, 0, 1}
	fmt.Println(nn.PredictDot(input, j))
	// Output: [1 1 1 1]
}

func ExampleAdd() {
	j := []float64{0, 0, 1, 1}
	fmt.Println(nn.Add(j, j))
	// Output: [0 0 2 2]
}

func ExampleSigmoidArray() {
	fmt.Println(nn.SigmoidArray([]float64{1, 2}))
	// Output: [0.7310585786300049 0.8807970779778823]
}

func ExampleSigmoidPrimeArray() {
	fmt.Println(nn.SigmoidPrimeArray([]float64{1, 2}))
	// Output: [0.19661193324148185 0.10499358540350662]
}

func ExampleWeights() {
	fmt.Println(nn.Weights(3))
	// Output: [0.20932057595923914 0.8810181760900249 0.32912010643698086]
}
