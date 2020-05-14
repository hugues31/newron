use crate::tensor::Tensor;
use std::fmt;

pub enum LearnableParams {
    Weights,
    Biases
}

pub trait Layer {
    fn forward(&mut self, input: Tensor, training: bool) -> Tensor;
    fn backward(&mut self, gradient: &Tensor) -> Tensor;
    fn get_info(&self) -> String;
    fn get_params_list(&self) -> Vec<LearnableParams>;
    fn get_param(&mut self, param: &LearnableParams) -> &mut Tensor;
    fn get_grad(&self, param: &LearnableParams) -> &Tensor;
}

impl fmt::Debug for dyn Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get_info())
    }
}