use crate::tensor::Tensor;

pub struct Layer {
    activation: String,
    unit: usize,
    pub input_size: usize,
    dropout: f32,
    pub weights: Tensor
}

impl Layer {
    pub fn new(activation: String, unit: usize, input_size: usize, dropout: f32) -> Layer {
        Layer {
            activation,
            unit,
            input_size,
            dropout,
            weights: Tensor::zero(vec![input_size, unit]) // replace zero with random
        }
    }
}
