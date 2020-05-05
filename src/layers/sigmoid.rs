use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct Sigmoid;

impl Layer for Sigmoid {
    fn get_info(&self) -> String {
        format!("Sigmoid Layer")
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| Sigmoid::sigmoid(x))
    }

    fn backward(&mut self, input: &Tensor, grad_output: Tensor) -> Tensor {
        let sigmoid_grad = input.map(|x| Sigmoid::sigmoid_prime(x));
        grad_output.mult_el(&sigmoid_grad)
    }
}

impl Sigmoid {
    pub fn sigmoid(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }

    pub fn sigmoid_prime(x: f64) -> f64 {
        Self::sigmoid(x) * (1. - Self::sigmoid(x))
    }
}
