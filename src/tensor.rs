// Implement basic tensor structure

use std::ops::{Add, Mul};
use std::fmt;


pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>
}

impl Tensor {
    /// Creates a new Tensor from `date` with the `shape` specified.
    pub fn new (data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor { data, shape }
    }

    /// Creates a Tensor filled with zeroes with the `shape` specified.
    pub fn zero(shape: Vec<usize>) -> Tensor {
        Tensor { data: vec![0.0; shape.iter().product()], shape }
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
            shape: vec![self.shape[1], self.shape[0]]
        }
    }

    /// get 2d positioned value in a 1d array
    fn get_value(&self, x: usize, y: usize) -> f64 {
        self.data[x * &self.shape[1] + y]
    }

    /// Get i-th row of the matrix. Return a new Tensor.
    /// Only for 2 dimensionals Tensor (matrix)
    pub fn get_row(&self, i: usize) -> Tensor {
        // TODO: add check for dimension
        Tensor {
            data: self.data[i*&self.shape[1]..(i+1)*&self.shape[1]].to_vec(),
            shape: vec![1, self.shape[1]]
        }
    }

    /// Creates a new Tensor where the function `f` is applied
    /// element-wise. Does not change the shape of tensor.
    pub fn map(&self, f: fn(f64) -> f64) -> Tensor {
        Tensor {
            data: self.data.to_vec().into_iter().map(f).collect(),
            shape: self.shape.to_vec()
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
                    println!("t += {} * {}", other.data[j], self.get_value(j, i));
                    t += other.data[j] * self.get_value(j, i);
                }

                sum_product.push(t);
                println!("{:?}", sum_product);
            }

            return Tensor::new(sum_product, vec![1, self.shape[1]]);
        }
        else {
            unimplemented!("Dot function not complete yet.")
        }
    }
}

// Implement addition for tensor
impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {

        if self.shape != other.shape {
            panic!("Could not add 2 tensors of different");
        }

        Tensor {
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect(),
            shape: self.shape
        }
    }
}


// impl<'a, 'b> Add<&'b Vector> for &'a Vector {
//     type Output = Vector;

//     fn add(self, other: &'b Vector) -> Vector {
//         Vector {
//             x: self.x + other.x,
//             y: self.y + other.y,
//         }
//     }
// }

// Implement multiplication for tensor
impl<'a, 'b> Mul<&'b Tensor> for  &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: &'b Tensor) -> Tensor {
        if self.shape.len() != self.shape.len() {
            panic!("Could not multiply tensors of different dimensions");
        }

        Tensor {
            data: match self.shape.len() {
                0 => vec![self.data[0] * other.data[0]],
                1 => self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect(),
                2 => {
                    // if # of rows of A is != # cols of B
                    if self.shape[0] != other.shape[1] {
                        panic!("Could not multiply tensors if # rows of A is different # cols of B.");
                    }
                    
                    // C = A*B = (m,n) * (n, k) = (m, k)

                    let m = self.shape[0];  // # rows of A
                    let n = self.shape[1];  // # cols of A
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
                _ => unimplemented!("Multiplication not implemented for Tensor of dimension > 2")
            },

            shape: match &self.shape.len() {
                0 => self.shape.to_vec(), // empty vec
                1 | 2 => vec![self.shape[0], other.shape[1]],
                _ => panic!("unsupported dimension*")
            }
        }
    }
}

// Implement equality test for tensor (PartialEq : a == b && b == a)
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

// Implement debug
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
         .field("data", &self.data)
         .field("shape", &self.shape)
         .finish()
    }
}