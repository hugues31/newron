/// The Sequential model is a linear stack of layers.

use crate::layer::Layer;
use crate::activation;

pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    /// Create a new empty Sequential model.
    pub fn new() -> Sequential {
        Sequential { layers: Vec::new::<>() }
    }

    /// Add a layer to the model
    pub fn add(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Use this function to train the model on x_train with target y_train.
    /// Set `verbose` to true to see debugging and training information.
    pub fn fit(&mut self, epochs: u32, verbose: bool) {
        let alpha = 0.002;

        for iteration in 0..epochs {
            let mut error = 0.0;

            // iterate through samples
            for i in 0..1 {
                // SGD implementation below
            }

            if verbose {
                if iteration % 10 == 9 {
                    println!("Error: {}", error);
                }
            }
        }


    }
}

