// Implement basic tensor structure

use crate::random::Rand;

use std::cmp;
use std::fmt;
use std::ops::{Add, Index, Mul, Sub, SubAssign};
use std::f64::consts::PI;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Creates a new Tensor from `data` with the `shape` specified.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor { data, shape }
    }

    /// Creates a Tensor filled with zeroes with the `shape` specified.
    pub fn zero(shape: Vec<usize>) -> Tensor {
        Tensor {
            data: vec![0.0; shape.iter().product()],
            shape,
        }
    }

    /// Creates a Tensor filled with ones with the `shape` specified.
    pub fn one(shape: Vec<usize>) -> Tensor {
        Tensor {
            data: vec![1.0; shape.iter().product()],
            shape,
        }
    }

    /// Creates a Tensor filled with uniformly distributed random values
    /// between -1 and +1 with the `shape` specified.
    pub fn random(shape: Vec<usize>, seed: u32) -> Tensor {
        let mut rng = Rand::new(seed);

        let number_values = shape.iter().product();
        let data: Vec<f64> = (0..number_values).map(|_| (rng.rand_float() - 0.5) * 2.0).collect();
        Tensor { data, shape }
    }

    /// Generates a Tensor filled with random values following a normal distribution
    /// with parameters mu and sigma specified (mean/stdev)
    pub fn random_normal(shape: Vec<usize>, mean: f64, stdev: f64, seed: u32) -> Tensor {
        // We use the Box-Muller method to generate random normal values
        // Formula: sqrt(-2*ln(rand()))*cos(2*Pi*rand()) * stdev + mean
        let mut rng = Rand::new(seed);

        let number_values = shape.iter().product();

        let data: Vec<f64> = (0..number_values).map(|_| (
            // formula
            ((-2.0 * rng.rand_float().ln()).sqrt() * (2.0 * PI * rng.rand_float()).cos()) * stdev + mean
        )).collect();

        Tensor { data, shape }
    }

    pub fn mask(shape: &Vec<usize>, prob: f32, seed: u32) -> Tensor {
        let mut result = vec![];
        let number_values = shape.iter().product();

        for i in 0..number_values {
            let t = (prob * number_values as f32) as usize;
            if i < t {
                result.push(0.0);
            } else {
                result.push(1.0);
            }
        }
        let mut rng = Rand::new(seed);
        rng.shuffle(&mut result[..]);

        Tensor::new(result, shape.to_vec())
    }

    /// Transpose matrix. Only for 2 dimensionals Tensor (matrix)
    pub fn transpose(&mut self) {
        // TODO: add check for dimension
        self.shape = vec![self.shape[1], self.shape[0]];
    }

    /// Creates new matrix based on the transposed `self` Tensor
    pub fn get_transpose(&self) -> Tensor {
        Tensor {
            data: self.data.to_vec(),
            shape: vec![self.shape[1], self.shape[0]],
        }
    }

    /// Compute the mean of the matrix along the `axis` specified.
    /// 0 = along the column, 1 = along the row
    pub fn get_mean(&self, axis: usize) -> Tensor {
        // TODO: refactor with same logic as get_max (or better!)
        let mut data = Vec::new();

        let other_axis = if axis == 1 { 0 } else { 1 };

        // the wording is not quite exact here
        // row and col is indeed the row and col for axis == 0
        // but row become col and col become when axis == 1
        for row in 0..self.shape[other_axis] {
            let mut acc = 0.0;
            for col in 0..self.shape[axis] {
                if axis == 0 {
                    acc += self.get_value(col, row);
                } else {
                    acc += self.get_value(row, col);
                }
                
            }
            data.push(acc / self.shape[axis] as f64);
        }

        let shape = vec![1, data.len()];
        Tensor::new(data, shape)
    }

    /// Get the Tensor containing max values along the `axis` specified
    /// 0 = along the column, 1 = along the row
    pub fn get_max(&self, axis: usize) -> Tensor {
        let mut data = Vec::new();

        if axis == 0 {
            for row in 0..self.shape[1] {
                let mut max = 0.0;
                for col in 0..self.shape[axis] {
                    let val = self.get_value(col, row);
                    if val > max {
                        max = val;
                    }
                }
                data.push(max);
            }
        } else {
            for col in 0..self.shape[0] {
                let mut max = 0.0;
                for row in 0..self.shape[axis] {
                    let val = self.get_value(col, row);
                    if val > max {
                        max = val;
                    }
                }
                data.push(max);
            }
        }

        let shape = if axis == 0 {vec![1, data.len()]} else {vec![data.len(), 1]};
        Tensor::new(data, shape)
    }

    /// Get 2d positioned value
    // 'data' is a flat array of f64
    fn get_value(&self, x: usize, y: usize) -> f64 {
        self.data[x * &self.shape[1] + y]
    }

    /// Get i-th row of the matrix. Return a new Tensor.
    /// Only for 2 dimensionals Tensor (matrix)
    pub fn get_row(&self, i: usize) -> Tensor {
        // TODO: add check for dimension
        Tensor {
            data: self.data[i * &self.shape[1]..(i + 1) * &self.shape[1]].to_vec(),
            shape: vec![1, self.shape[1]],
        }
    }

    /// Get all rows from a vector containing indices
    pub fn get_rows(&self, indices: &[usize]) -> Tensor {
        let mut data = Vec::new();
        for i in indices {
            data.extend(self.get_row(*i).data.iter());
        }
        Tensor {
            data,
            shape: vec![indices.len(), self.shape[1]],
        }
    }

    /// Creates a new Tensor where the function `f` is applied
    /// element-wise. Does not change the shape of tensor.
    pub fn map(&self, f: fn(f64) -> f64) -> Tensor {
        Tensor {
            data: self.data.to_vec().into_iter().map(f).collect(),
            shape: self.shape.to_vec(),
        }
    }

    /// Dot product much like the numpy implementation
    /// Dot product of two arrays. Specifically :
    /// If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation). TODO
    /// If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred. TODO
    /// If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred. TODO
    /// If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b. OK
    /// If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b. TODO
    pub fn dot(&self, other: &Tensor) -> Tensor {
        if self.shape.len() > 1 && other.shape[0] == 1 {
            // Sum product over the last axis of self and other
            let mut sum_product = Vec::new();

            for i in 0..self.shape[1] {
                let mut t = 0.0;
                // TODO improve with N dimensions for &self (current implementation works only for Matrix)
                for j in 0..other.shape[1] {
                    t += other.data[j] * self.get_value(j, i);
                }
                sum_product.push(t);
            }

            return Tensor::new(sum_product, vec![1, self.shape[1]]);
        } else if self.shape[1] == other.shape[0] {
            return self * other;
        } else {
            unimplemented!("Dot function not complete yet dot({:?},{:?}).", self, other)
        }
    }

    /// Element-wise multiplication (or Hadamard product)
    pub fn mult_el(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);

        Tensor::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect(),
            self.shape.to_vec(),
        )
    }
}

// Implement addition for tensor references
impl<'a, 'b> Add<&'b Tensor> for &'b Tensor {
    type Output = Tensor;

    fn add(self, other: &'b Tensor) -> Tensor {
        if self.shape.len() > 1 && other.shape.len() > 1 {
            if self.shape[1] != other.shape[1] {
                panic!("Could not add 2 tensors of different. {:?} + {:?}", self, other);
            }          
        }

        let mut data = Vec::new();
        for i in 0..cmp::max(self.data.len(), other.data.len()) {
            let a = self.data[i % self.data.len()];
            let b = other.data[i % other.data.len()];
            data.push(a + b);
        }

        Tensor {
            data,
            shape: self.shape.to_vec(),
        }
    }
}

// Implement addition for tensor
impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        &self + &other
    }
}

// Implement substraction for tensor
impl<'a, 'b> Sub<&'b Tensor> for &'b Tensor {
    type Output = Tensor;

    fn sub(self, other: &'b Tensor) -> Tensor {
        if self.shape.len() > 1 && other.shape.len() > 1 {
            if self.shape[1] != other.shape[1] {
                panic!("Could not substract 2 tensors of different. {:?} - {:?}", self, other);
            }          
        }

        let mut data = Vec::new();
        for i in 0..cmp::max(self.data.len(), other.data.len()) {
            let a = self.data[i % self.data.len()];
            let b = other.data[i % other.data.len()];
            data.push(a - b);
        }

        Tensor {
            data,
            shape: self.shape.to_vec(),
        }
    }
}

// Implement substract assignation for tensor
// E.g : tensor -= other_tensor
impl SubAssign for Tensor {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
            shape: self.shape.to_vec(),
        };
    }
}

// Implement multiplication for tensor
impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: &'b Tensor) -> Tensor {
        if self.shape.len() != self.shape.len() {
            panic!("Could not multiply tensors of different dimensions");
        }

        Tensor {
            data: match self.shape.len() {
                0 => vec![self.data[0] * other.data[0]],
                1 => self
                    .data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(a, b)| a * b)
                    .collect(),
                2 => {
                    // if # of cols of A is != # rows of B
                    if self.shape[1] != other.shape[0] {
                        panic!("Could not multiply matrix if # cols of A is different # rows of B.\nA: {:?}\nB: {:?}", self, other);
                    }

                    // C = A*B = (m,n) * (n, k) = (m, k)

                    let m = self.shape[0]; // # rows of A
                    let n = self.shape[1]; // # cols of A
                    let k = other.shape[1]; // # cols of B

                    // let mut c: Vec<f64> = Vec::with_capacity(m * k);
                    let mut c: Vec<f64> = Vec::new();
                    for i in 0..m {
                        for j in 0..k {
                            let mut c_ij = 0.0;
                            for s in 0..n {
                                c_ij = c_ij + &self.get_value(i, s) * &other.get_value(s, j);
                            }
                            c.push(c_ij);
                        }
                    }

                    c
                }
                _ => unimplemented!("Multiplication not implemented for Tensor of dimension > 2"),
            },

            shape: match &self.shape.len() {
                0 => self.shape.to_vec(), // empty vec
                1 | 2 => vec![self.shape[0], other.shape[1]],
                _ => panic!("unsupported dimension*"),
            },
        }
    }
}

// Implement multiplication for tensor with scalar (f64)
// Multiplication is element-wise in this case
impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, other: f64) -> Tensor {
        Tensor {
            data: self.data.iter().map(|a| a * other).collect(),
            shape: self.shape.to_vec(),
        }
    }
}

impl Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        Tensor {
            data: other.data.iter().map(|a| a * self).collect(),
            shape: other.shape.to_vec(),
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, other: f32) -> Tensor {
        Tensor {
            data: self.data.iter().map(|a| a * other as f64).collect(),
            shape: self.shape.to_vec(),
        }
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        Tensor {
            data: other.data.iter().map(|a| a * self as f64).collect(),
            shape: other.shape.to_vec(),
        }
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        &self * &other
    }
}

impl Index<usize> for Tensor {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.data[index]
    }
}

// Implement equality test for tensor (PartialEq : a == b && b == a)
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

// Implement Debug
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:?}\n", self, self.shape)
    }
}

// Implement Display
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 0D: "3.14"
        // 1D: "[1.0, 2.0, 3.0]"
        // 2D: "|1.0, 2.0|
        //      |3.0, 4.0|"
        // ND: "?"

        match self.shape.len() {
            0 => write!(f, "{}", self.data[0]),
            1 => {
                let mut result = String::new();
                for el in &self.data {
                    result += &el.to_string();
                }
                write!(f, "[{}]", result)
            }
            2 => {
                // maximum of 4 decimals are shown
                let decimals = 4;
                let mut result = String::from("\n");
                for row in 0..self.shape[0] {
                    result += "|";
                    for col in 0..self.shape[1] {
                        let mut value = self.get_value(row, col).to_string();
                        
                        if value.len() > decimals {
                            value = value[0..decimals].to_string();
                        }
                        else {
                            value = value.to_string() + &" ".repeat(decimals - &value.len());
                        }
                        result += &(value + "  ")
                    }
                    result += "|\n";
                }
                write!(f, "{}", result)
            }
            _ => unimplemented!(),
        }
    }
}
