use crate::tensor::Tensor;
use std::fmt;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&mut self, input: &Tensor, grad_output: Tensor) -> Tensor;
    fn get_info(&self) -> String;
}

impl fmt::Debug for dyn Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get_info())
    }
}