/// The Sequential model is a linear stack of layers.
use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::dataset::{Dataset, RowType, ColumnType};

pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
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
    pub fn add<T: 'static +  Layer>(&mut self, layer: T) {
        self.layers.push(Box::new(layer));
    }

    // Return the list of layer outputs given an input
    fn forward_propagation(&mut self, input: &Tensor, train: bool) -> Vec<Tensor> {
        let mut activations = Vec::new();

        for layer in &self.layers {

        }

        activations
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
