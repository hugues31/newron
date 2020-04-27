struct ReLU;

impl Layer for ReLU {
    fn forward(input: Tensor) -> Tensor {
        input.map(|x| x.max(0))
    }

    fn backward(input: Tensor, grad_output: Tensor) -> Tensor {
        input.map(|x| if x < 0 { 0 } else { x })
    }
}