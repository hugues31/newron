/// The Sequential model is a linear stack of layers.

use crate::layer::Layer;
use crate::Matrix;
use crate::activation;

pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential { layers: Vec::new::<>() }
    }

    // Add a layer to the model
    pub fn add(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    // Fit the model
    pub fn fit(&mut self, x_train: Matrix, y_train: Matrix, epochs: u32, verbose: bool) {
        let alpha = 0.2;

        for iteration in 0..epochs {
            let mut error = 0.0;

            // iterate through samples
            for i in 0..self.layers.first().unwrap().weights.shape().0 {
                // SGD
                let layer_0 = x_train.row(i);
                let layer_1 = (layer_0*&self.layers[0].weights).map(activation::relu);
                let layer_2 = &layer_1 * &self.layers[1].weights;
                error += (&layer_2 - y_train.row(i))[0].powi(2);
                let layer_2_delta = &layer_2 - y_train.row(i);
    
                let layer_1_delta_dot = &layer_2_delta * &self.layers[1].weights.transpose();
                let layer_1_delta = layer_1_delta_dot.component_mul(&layer_1.map(activation::relu2deriv));
      
                self.layers[1].weights -= alpha * layer_1.transpose() * &layer_2_delta;
                self.layers[0].weights -= alpha * layer_0.transpose() * &layer_1_delta;
            }

            if verbose {
                if iteration % 10 == 9 {
                    println!("Error: {}", error);
                }
            }
        }


    }
}

