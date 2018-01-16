package main

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"strconv"
	"strings"
	"time"

	m "../matrix"
	n "../network"
)

type Label struct {
	X m.Matrix
	Y m.Matrix
}

func ReadFile(filePath string) []byte {
	byteContent, err := ioutil.ReadFile(filePath)
	if err != nil {
		panic(err)
	}

	return byteContent
}

func DataByteToLabel(byteContent []byte, outputNumber int) []Label {
	conetent := strings.TrimSuffix(string(byteContent), "\n")
	lines := strings.Split(conetent, "\n")

	data := []Label{}

	for _, line := range lines {
		labels := strings.Split(line, ",")
		row := Label{}
		row.Y.Create(1, outputNumber)

		xs := []float64{}
		for i := 1; i < len(labels)-1; i++ {
			f, err := strconv.ParseFloat(labels[i], 64)
			if err != nil {
				panic(err)
			}
			//fmt.Print(f, " ")
			xs = append(xs, f)
		}

		row.X = m.Convert(xs)

		if strings.Compare(labels[len(labels)-1], "Iris-setosa") == 0 {
			row.Y.Matrix[0][0] = 1.0
		} else if strings.Compare(labels[len(labels)-1], "Iris-versicolor") == 0 {
			row.Y.Matrix[0][1] = 1.0
		} else if strings.Compare(labels[len(labels)-1], "Iris-virginica") == 0 {
			row.Y.Matrix[0][2] = 1.0
		}

		//fmt.Print("\n")
		data = append(data, row)
	}

	return data
}

func main() {
	fmt.Println("Hello")

	rand.Seed(time.Now().UnixNano())

	var a m.Matrix
	var b m.Matrix

	a.Create(1, 4)
	a.Random()
	a.Print()
	b.Create(1, 2)
	b.Matrix[0][0] = 0
	b.Matrix[0][1] = 1

	b.Print()

	//5.1,3.5,1.4,0.2,Iris-setosa

	bytes := ReadFile("./iris.data")
	la := DataByteToLabel(bytes, 3)

	for _, el := range la {
		el.X.Print()
		el.Y.Print()
	}

	var net n.Network
	net.Init(3, []int{256, 256}, 3)
	net.Print()
	//	fmt.Println("--------------------------------------")
	//	net.Predict(a).Print()
	//	net.Loss(a, b).Print()

	fmt.Println("----------------------------------")
	for i := 0; i < 100; i++ {
		for _, el := range la {
			//n.Softmax(net.Predict(el.X)).Print()

			//net.Loss(a, b).Print()
			grad := net.Gradient(el.X, el.Y)
			//grad.Print()

			net.Update(grad, 0.0001)

		}
	}
	fmt.Println("----------------------------------")

	//net.Print()
	for _, el := range la {
		fmt.Println("----------------------------------")
		n.Softmax(net.Predict(el.X)).Print()
		el.Y.Print()
	}

	net.Save("example")

	fmt.Println("----------------------------------")
	var net2 n.Network
	net2.Load("example")
	//net2.Print()
	//	a.Mul(b).Print()
	//	b.Transpose().Print()
}
