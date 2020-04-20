/// The Sequential model is a linear stack of layers.
use crate::layer::Layer;
use crate::tensor::Tensor;

pub struct Sequential {
    pub layers: Vec<Layer>,
    weights: Vec<Tensor>,
    seed: u32
}

impl Sequential {
    /// Create a new empty Sequential model.
    pub fn new() -> Sequential {
        Sequential {
            layers: vec![],
            weights: vec![],
	    seed: 0
        }
    }

    // Immutable access.
    fn get_seed(&self) -> &u32 {
        &self.seed
    }

    // Mutable access.
    pub fn set_seed(&mut self, s: u32) {
        self.seed = s;
    }

    /// Add a layer to the model
    pub fn add(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    // Return the list of layer outputs given an input
    fn forward_propagation(&self, input: &Tensor, train: bool) -> Vec<Tensor> {
        // Forward propagation
        let mut outputs: Vec<Tensor> = Vec::new();
        // ouput of the first layer is the training sample...
        outputs.push(input.get_transpose());
        for (i, w) in self.weights.iter().enumerate() {
            if train {
                let dropout = &self.layers[i].dropout;
                let seed_layer_i = &self.seed + i as u32;// + input_layer_i;
                let dropout_mask = Tensor::mask(&w.shape, *dropout, seed_layer_i);
                let output = (&((1.0 / (1.0 - dropout)) * w.mult_el(&dropout_mask))
                    * outputs.last().unwrap())
                .map(&self.layers[i].activation.activation());

                outputs.push(output);
            } else {
                let output =
                    (w * &outputs.last().unwrap()).map(&self.layers[i].activation.activation());
                outputs.push(output);
            }
        }

        outputs
    }

    /// Use this function to train the model on x_train with target y_train.
    /// Set `verbose` to true to see debugging and training information.
    pub fn fit(&mut self, x_train: &Tensor, y_train: &Tensor, epochs: u32, verbose: bool) {
        let alpha = 0.02;

        // Initialize weights with random values
        self.weights.clear();
        for i in 0..self.layers.len() {
            let unit = self.layers[i].unit;
            let input_size = if i == 0 {
                x_train.shape[1]
            } else {
                self.layers[i - 1].unit
            };
            self.weights.push(Tensor::random(vec![unit, input_size], self.seed));
        }

        for iteration in 0..epochs {
            let mut error = 0.0;

            // iterate through samples
            for i in 0..x_train.shape[0] {
                // SGD implementation below

                // Forward propagation
                let outputs = &self.forward_propagation(&x_train.get_row(i), true);

                // Compute error TODO: compute error on verbose only (or early stopping)
                error += (outputs.last().unwrap() - &y_train.get_row(i))[0].powi(2);

                // Compute backward pass
                let mut gradients: Vec<Tensor> = Vec::new();
                // First gradient (delta L)
                let gradient = outputs.last().unwrap() - &y_train.get_row(i);
                gradients.push(
                    gradient.mult_el(
                        &(self.weights.last().unwrap() * &outputs[outputs.len() - 2])
                            .map(&self.layers.last().unwrap().activation.deriv_activation()),
                    ),
                );

                // Other gradients (delta i)
                for (i, w) in self.weights.iter().skip(1).rev().enumerate() {
                    let left_gradient = &w.get_transpose() * &gradients.last().unwrap();
                    let right_gradient = (&self.weights[self.weights.len() - 2 - i]
                        * &outputs[outputs.len() - 3 - i])
                        .map(
                            &self.layers[self.layers.len() - 2 - i]
                                .activation
                                .deriv_activation(),
                        );
                    let gradient = left_gradient.mult_el(&right_gradient);
                    gradients.push(gradient);
                }

                // Weight update
                for (i, w) in self.weights.iter_mut().enumerate() {
                    *w -=
                        alpha * (&gradients[gradients.len() - 1 - i] * &outputs[i].get_transpose());
                }
            }
            if verbose {
                if iteration % 10 == 9 {
                    println!("Error: {}", error);
                }
            }
        }
    }

    pub fn predict(&self, input: &Tensor) -> Tensor {
        // The output of the network is the last layer output

        match self.forward_propagation(input, false).last() {
            Some(x) => x.clone(),
            None => panic!("No prediction."),
        }
    }
}
