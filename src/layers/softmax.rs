use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct Softmax;

impl Layer for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        // we use stable softmax instead of classic softmax
        // for computational stability
        // z -= np.max(z)
        // sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
        // return sm
        let max_input = input.get_max(1);
        let normalized_input = &max_input - input;
        let numerator = normalized_input.map(|x| x.exp()).get_transpose();
        // let denominator = 
        numerator
    }

    fn backward(&mut self, input: &Tensor, grad_output: Tensor) -> Tensor {
        let relu_grad = input.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        grad_output.mult_el(&relu_grad)
    }
}