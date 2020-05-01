use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct Softmax;

impl Layer for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        // we use stable softmax instead of classic softmax
        // for computational stability

        let normalized_input = input.normalize_rows();
        let numerator = normalized_input.map(|x| x.exp());
        let denominator = normalized_input.map(|x| x.exp()).get_sum(1);
        numerator / denominator
    }

    fn backward(&mut self, input: &Tensor, grad_output: Tensor) -> Tensor {
        unimplemented!()
    }
}