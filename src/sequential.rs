/// The Sequential model is a linear stack of layers.

use crate::layer::Layer;
use crate::activation;
use crate::tensor::Tensor;

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
    pub fn fit(&mut self, x_train: Tensor, y_train: Tensor, epochs: u32, verbose: bool) {
        let alpha = 0.002;

        for iteration in 0..epochs {
            let mut error = 0.0;

            // iterate through samples
            for i in 0..x_train.shape[0] {
                // SGD implementation below

                // forward propagation
                let mut outputs: Vec<Tensor> = Vec::new();
                outputs.push(x_train.get_row(i));

                for l in &self.layers {
                    println!("a: {:?}, b: {:?}", outputs.last().unwrap(), &l.weights);
                    let layer_l =  l.weights.dot(&outputs.last().unwrap()).map(activation::relu);
                    outputs.push(layer_l);
                }

                // error += (outputs.last().unwrap() - y_train.row(i))[0].powi(2);
                // // compute gradient
                // let mut gradients: Vec<Matrix> = Vec::new();
                
                // gradients.push(outputs.last().unwrap() - y_train.rows(i, 1));
                
                // for (j, l) in self.layers.iter().skip(0).rev().enumerate() {
                //     let gradient_dot = gradients.last().unwrap() * l.weights.transpose();
                //     let gradient = gradient_dot.component_mul(&outputs[outputs.len() -2 - j].map(activation::relu2deriv));
                //     gradients.push(gradient);
                // }

                // // back propagation
                // for (i, l) in self.layers.iter_mut().rev().enumerate() {
                //     l.weights -= alpha * &outputs[outputs.len() - i -2].transpose() * &gradients[i];
                // }
            }

            if verbose {
                if iteration % 10 == 9 {
                    println!("Error: {}", error);
                }
            }
        }
    }
}

