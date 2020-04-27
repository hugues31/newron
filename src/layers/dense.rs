use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct Dense {
    weights: Tensor,
    biases: Tensor
}

impl Dense {
    pub fn new(input_units: usize, output_units: usize) -> Dense {
        unimplemented!();
    }
}

impl Layer for Dense {
    fn forward(self, input: Tensor) -> Tensor {
        input.dot(&self.weights) + self.biases
    }

    fn backward(self, input: Tensor, grad_output: Tensor) -> Tensor {
        unimplemented!();
    }
}