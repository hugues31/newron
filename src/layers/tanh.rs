use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct Tanh;

impl Layer for Tanh {
    fn get_info(&self) -> String {
        format!("Tanh Layer")
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| Tanh::tanh(x))
    }

    fn backward(&mut self, input: &Tensor, grad_output: Tensor) -> Tensor {
        let tanh_grad = input.map(|x| Tanh::tanh_prime(x));
        println!("grad tanh: \n{}", grad_output.mult_el(&tanh_grad));
        grad_output.mult_el(&tanh_grad)
    }
}

impl Tanh {
    fn tanh(x: f64) -> f64 {
        (2.0 / (1.0 + (-2.0 * x).exp())) - 1.0
    }

    fn tanh_prime(x: f64) -> f64 {
        1.0 - Self::tanh(x).powi(2)
    }
}
