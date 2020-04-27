/// The Sequential model is a linear stack of layers.
use crate::layers;
use crate::tensor::Tensor;
use crate::dataset::{Dataset, RowType, ColumnType};

pub struct Sequential {
    pub layers: Vec<Layer>,
    seed: u32,
}

impl Sequential {
    /// Create a new empty Sequential model.
    pub fn new() -> Sequential {
        Sequential {
            layers: vec![],
            seed: 0,
        }
    }

    /// Seed the random number generator
    pub fn set_seed(&mut self, s: u32) {
        self.seed = s;
    }

    /// Add a layer to the model
    pub fn add(&mut self, layer: Layer) {
        unimplemented!();
    }

    // Return the list of layer outputs given an input
    fn forward_propagation(&mut self, input: &Tensor, train: bool) -> Vec<Tensor> {
        unimplemented!();
    }

    /// Use this function to train the model on x_train with target y_train.
    /// Set `verbose` to true to see debugging and training information.
    pub fn fit(&mut self, dataset: &Dataset, epochs: u32, verbose: bool) {
        unimplemented!();
    }

    pub fn predict(&mut self, input: &Vec<f64>) -> Tensor {
        unimplemented!();
    }
}
