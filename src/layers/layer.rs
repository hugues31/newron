use crate::tensor::Tensor;

trait Layer {
    fn forward(input: Tensor) -> Tensor;
    fn backward(input: Tensor, grad_output: Tensor) -> Tensor;
}

