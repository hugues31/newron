use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct ReLU;

impl Layer for ReLU {
    fn forward(self, input: Tensor) -> Tensor {
        input.map(|x| x.max(0.0))
    }

    fn backward(self, input: Tensor, grad_output: Tensor) -> Tensor {
        input.map(|x| if x < 0.0 { 0.0 } else { x })
    }
}