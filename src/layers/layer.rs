use crate::tensor::Tensor;

pub trait Layer {
    fn forward(self, input: Tensor) -> Tensor;
    fn backward(self, input: Tensor, grad_output: Tensor) -> Tensor;
}

