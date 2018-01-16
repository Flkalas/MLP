package matrix

import (
	"fmt"
	"math/rand"
)

type Matrix struct {
	Col int
	Row int

	Matrix [][]float64
}

func (m *Matrix) Create(row int, col int) {
	m.Col = col
	m.Row = row

	for i := 0; i < row; i++ {
		row := make([]float64, col)
		m.Matrix = append(m.Matrix, row)
	}
}

func (m *Matrix) Random() {
	for i, _ := range m.Matrix {
		for j, _ := range m.Matrix[i] {
			m.Matrix[i][j] = rand.NormFloat64()
		}
	}
}

func (a Matrix) Add(b Matrix) Matrix {
	var c Matrix

	if (a.Col != b.Col) || (a.Row != b.Row) {
		return c
	}

	c.Create(a.Row, a.Col)

	for i := 0; i < a.Row; i++ {
		for j := 0; j < a.Col; j++ {
			c.Matrix[i][j] = a.Matrix[i][j] + b.Matrix[i][j]
		}
	}

	return c
}

func (a Matrix) Sub(b Matrix) Matrix {
	var c Matrix

	if (a.Col != b.Col) || (a.Row != b.Row) {
		return c
	}

	c.Create(a.Row, a.Col)

	for i := 0; i < a.Row; i++ {
		for j := 0; j < a.Col; j++ {
			c.Matrix[i][j] = a.Matrix[i][j] - b.Matrix[i][j]
		}
	}

	return c
}

func (a Matrix) Mul(b Matrix) Matrix {
	var c Matrix

	if a.Col != b.Row {
		return c
	}

	c.Create(a.Row, b.Col)

	for i := 0; i < a.Row; i++ {
		for j := 0; j < b.Col; j++ {
			for k := 0; k < b.Row; k++ {
				c.Matrix[i][j] += a.Matrix[i][k] * b.Matrix[k][j]
			}
		}
	}

	return c
}

func (a Matrix) MulConst(k float64) Matrix {
	for i := 0; i < a.Row; i++ {
		for j := 0; j < a.Col; j++ {
			a.Matrix[i][j] *= k
		}
	}

	return a
}

func (m Matrix) Transpose() Matrix {
	var a_t Matrix
	a_t.Create(m.Col, m.Row)

	for i, _ := range m.Matrix {
		for j, _ := range m.Matrix[0] {
			a_t.Matrix[j][i] = m.Matrix[i][j]
		}
	}

	return a_t
}

func (m Matrix) Print() {
	for i := range m.Matrix {
		fmt.Printf("Row: %v ", i)
		fmt.Println(m.Matrix[i])
	}
}
