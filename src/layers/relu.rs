use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct ReLU;

impl Layer for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, input: &Tensor, grad_output: Tensor) -> Tensor {
        let relu_grad = input.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        grad_output.mult_el(&relu_grad)
    }
}