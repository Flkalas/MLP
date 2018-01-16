package network

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io/ioutil"
	"math"

	m "../matrix"
)

type Network struct {
	NumberLayer int

	Input  int
	Hidden []int
	Output int

	AffineLayer  []Affine
	ReluLayer    []Relu
	SoftmaxLayer SoftmaxWithLoss
}

func getConnectedMatrix(input int, hidden []int, output int) []int {
	n := []int{input}
	for _, eachHidden := range hidden {
		n = append(n, eachHidden)
	}
	n = append(n, output)
	return n
}

func (n *Network) Init(input int, hidden []int, output int) {
	n.Input = input
	n.Output = output
	n.Hidden = hidden

	n.NumberLayer = len(hidden) + 2

	connMatrix := getConnectedMatrix(input, hidden, output)

	for i := 0; i < n.NumberLayer-1; i++ {
		var a Affine

		var a_w m.Matrix
		a_w.Create(connMatrix[i], connMatrix[i+1])
		a_w.Random()
		a.Weight = a_w

		var a_b m.Matrix
		a_b.Create(1, connMatrix[i+1])
		a_b.Random()
		a.Bias = a_b

		n.AffineLayer = append(n.AffineLayer, a)

		if i < n.NumberLayer-2 {
			var r Relu
			n.ReluLayer = append(n.ReluLayer, r)
		}
	}

	var s SoftmaxWithLoss
	n.SoftmaxLayer = s
}

func (n *Network) Predict(input m.Matrix) m.Matrix {
	for i := 0; i < n.NumberLayer-1; i++ {
		input = n.AffineLayer[i].Forward(input)

		if i < n.NumberLayer-2 {
			input = n.ReluLayer[i].Forward(input)
		}
	}

	return input
}

func (n *Network) Loss(input m.Matrix, train m.Matrix) m.Matrix {
	y := n.Predict(input)
	return n.SoftmaxLayer.Forward(y, train)
}

func (n *Network) Gradient(input m.Matrix, train m.Matrix) Network {
	n.Loss(input, train)

	var dOut m.Matrix
	dOut.Create(1, 1)
	dOut.Matrix[0][0] = 1.0
	dOut = n.SoftmaxLayer.Backward(dOut)

	for i := 0; i < n.NumberLayer-1; i++ {
		if n.NumberLayer-2-i < n.NumberLayer-2 {
			dOut = n.ReluLayer[n.NumberLayer-2-i].Backward(dOut)
		}
		dOut = n.AffineLayer[n.NumberLayer-2-i].Backward(dOut)
	}

	var net Network
	net.Init(n.Input, n.Hidden, n.Output)

	for i := 0; i < n.NumberLayer-1; i++ {
		net.AffineLayer[i].Weight = n.AffineLayer[i].DeltaWeight
		net.AffineLayer[i].Bias = n.AffineLayer[i].DeltaBias
	}

	return net
}

func (n *Network) Update(grad Network, learningRate float64) {
	for i := 0; i < n.NumberLayer-1; i++ {
		n.AffineLayer[i].Weight = n.AffineLayer[i].Weight.Sub(grad.AffineLayer[i].Weight.MulConst(learningRate))
		n.AffineLayer[i].Bias = n.AffineLayer[i].Bias.Sub(grad.AffineLayer[i].Bias.MulConst(learningRate))
	}
}

func (n Network) Print() {
	for i, eachA := range n.AffineLayer {
		fmt.Println("Weight:", i)
		eachA.Weight.Print()
		fmt.Println("Bias:", i)
		eachA.Bias.Print()
	}
}

func (n Network) Save(fileName string) {
	gob.Register(Network{})

	var binanyBuffer bytes.Buffer
	encoder := gob.NewEncoder(&binanyBuffer)
	err := encoder.Encode(n)
	check(err)

	err = ioutil.WriteFile(fileName, binanyBuffer.Bytes(), 0644)
	check(err)
}

func (n *Network) Load(fileName string) {
	var binanyBuffer bytes.Buffer
	dat, err := ioutil.ReadFile(fileName)
	check(err)

	binanyBuffer.Write(dat)

	decoder := gob.NewDecoder(&binanyBuffer)
	err = decoder.Decode(n)
	check(err)
}

type Affine struct {
	Weight m.Matrix
	Bias   m.Matrix

	X           m.Matrix
	DeltaWeight m.Matrix
	DeltaBias   m.Matrix
}

func (a *Affine) Forward(x m.Matrix) m.Matrix {
	a.X = x
	out := x.Mul(a.Weight).Add(a.Bias)
	return out
}

func (a *Affine) Backward(dOut m.Matrix) m.Matrix {
	dx := dOut.Mul(a.Weight.Transpose())
	a.DeltaWeight = a.X.Transpose().Mul(dOut)
	a.DeltaBias = dOut
	return dx
}

type Relu struct {
	Mask [][]bool
}

func (r *Relu) Forward(x m.Matrix) m.Matrix {
	var out m.Matrix
	r.Mask = [][]bool{}

	out.Create(x.Row, x.Col)

	for i, _ := range x.Matrix {
		row := []bool{}
		for j, _ := range x.Matrix[i] {
			row = append(row, (x.Matrix[i][j] <= 0))
			out.Matrix[i][j] = math.Max(x.Matrix[i][j], 0.0)
		}
		r.Mask = append(r.Mask, row)
	}

	return out
}

func (r Relu) Backward(dout m.Matrix) m.Matrix {
	dx := dout

	for i, _ := range dx.Matrix {
		for j, _ := range dx.Matrix[i] {
			if r.Mask[i][j] {
				dx.Matrix[i][j] = 0.0
			}
		}
	}

	return dx
}

type SoftmaxWithLoss struct {
	Loss m.Matrix
	Y    m.Matrix
	T    m.Matrix
}

func (s *SoftmaxWithLoss) Forward(x m.Matrix, t m.Matrix) m.Matrix {
	s.T = t
	s.Y = Softmax(x)
	s.Loss = CrossEntropyError(s.Y, s.T)
	return s.Loss
}

func (s *SoftmaxWithLoss) Backward(dOut m.Matrix) m.Matrix {
	dx := s.Y.Sub(s.T)
	return dx
}

func Softmax(a m.Matrix) m.Matrix {
	maxA := []float64{}
	for j, _ := range a.Matrix {
		maxA = append(maxA, a.Matrix[j][0])
		for i := 1; i < len(a.Matrix[j]); i++ {
			maxA[j] = math.Max(maxA[j], a.Matrix[j][i])
		}
	}

	sumExpA := []float64{}

	var expA m.Matrix
	expA.Create(a.Row, a.Col)
	for j, _ := range a.Matrix {
		sumExpA = append(sumExpA, 0.0)
		for i := 0; i < len(a.Matrix[j]); i++ {
			expA.Matrix[j][i] = math.Exp(a.Matrix[j][i] - maxA[j])
			sumExpA[j] += expA.Matrix[j][i]
		}
	}

	var y m.Matrix
	y.Create(a.Row, a.Col)
	for j, _ := range a.Matrix {
		for i := 0; i < len(a.Matrix[j]); i++ {
			y.Matrix[j][i] = expA.Matrix[j][i] / sumExpA[j]
		}
	}

	return y
}

func CrossEntropyError(output m.Matrix, trainOuput m.Matrix) m.Matrix {
	var out m.Matrix
	out.Create(output.Row, 1)
	for i, _ := range output.Matrix {
		sum := 0.0
		for j, _ := range output.Matrix[i] {
			sum += trainOuput.Matrix[i][j] * math.Log(output.Matrix[i][j]+math.SmallestNonzeroFloat64)
		}
		out.Matrix[i][0] = -sum
	}
	return out
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}
