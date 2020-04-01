use crate::Matrix;

pub struct Layer {
    activation: String,
    unit: usize,
    input_size: usize,
    dropout: f32,
    pub weights: Matrix
}

impl Layer {
    pub fn new(activation: String, unit: usize, input_size: usize, dropout: f32) -> Layer {
        Layer {
            activation,
            unit,
            input_size,
            dropout,
            weights: Matrix::new_random(input_size, unit).map(|x| 2.0 * x - 1.0)
        }
    }
}
