use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&mut self, input: &Tensor, grad_output: Tensor) -> Tensor;
}

