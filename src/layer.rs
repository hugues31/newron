use crate::activation::Activation;

pub struct Layer {
    pub activation: Activation,
    pub unit: usize,
    pub dropout: f32,
    pub seed: u32
}

impl Layer {
    pub fn new(activation: Activation, unit: usize, dropout: f32, seed: u32) -> Layer {
        Layer {
            activation,
            unit,
            dropout,
            seed
        }
    }
}
