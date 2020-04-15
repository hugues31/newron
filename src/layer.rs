pub struct Layer {
    activation: String,
    pub unit: usize,
    dropout: f32
}

impl Layer {
    pub fn new(activation: String, unit: usize, dropout: f32) -> Layer {
        Layer {
            activation,
            unit,
            dropout
        }
    }
}
