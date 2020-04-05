// Implement basic tensor structure

use std::ops::{Add, Mul};
use std::fmt;

pub struct Tensor {
    pub data: Vec<f64>,
    shape: Vec<usize>,
    creators: Vec<&Tensor>
}

// get 2d positioned value in a 1d array
fn get_value(tensor: &Tensor, x: usize, y: usize) -> f64 {
    tensor.data[x * tensor.shape[1] + y]
}

impl Tensor {
    pub fn new (data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor { data, shape, creators: Vec::new() }
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
            shape: self.shape,
            creators: vec![&self, other]
        }
    }
}

// Implement multiplication for tensor
impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
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
                                c_ij = c_ij + (get_value(&self, i, s) * get_value(&other, s, j));
                            }
                            c.push(c_ij);
                        }
                    }

                    c
                }
                _ => unimplemented!("Multiplication not implemented for Tensor of dimension > 2")
            },

            shape: match &self.shape.len() {
                0 => self.shape, // empty vec
                1 | 2 => vec![self.shape[0], other.shape[1]],
                _ => panic!("unsupported dimension*")
            },

            creators: Vec::new()
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