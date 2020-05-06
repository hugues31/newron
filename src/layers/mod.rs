pub mod layer;
pub mod relu;
pub mod tanh;
pub mod dense;
pub mod softmax;


pub enum LayerEnum {
    Dense {input_units: usize, output_units: usize},
    ReLU,
    Softmax,
    TanH
}