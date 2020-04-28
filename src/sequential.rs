/// The Sequential model is a linear stack of layers.
use std::cmp;

use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::dataset::{Dataset, RowType, ColumnType};
use crate::random::Rand;

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
        // Compute activations of all network layers by applying them sequentially.
        // Return a list of activations for each layer.

        let mut activations: Vec<Tensor> = Vec::new();
        
        // First propagation with the input
        activations.push(self.layers.first().unwrap().forward(&input));

        // Next propagations with the last propagated values
        for layer in self.layers.iter().skip(1) {
            activations.push(layer.forward(activations.last().unwrap()));
        }

        assert_eq!(activations.len(), self.layers.len());
        activations
    }

    /// Use this function to train the model on x_train with target y_train.
    /// Set `verbose` to true to see debugging and training information.
    pub fn fit(&mut self, dataset: &Dataset, epochs: u32, verbose: bool) {
        let x_train = dataset.get_tensor(RowType::Train, ColumnType::Feature); 
        let y_train = dataset.get_tensor(RowType::Train, ColumnType::Target);

        // TODO: Check model architecture (input_unit == x_train.len(),
        // output_unit_l == input_unit_l+1, output_unit_l_n = y_train.len()) and display message here
        
        // auto batch size : TODO improve it
        let batch_size = cmp::min(x_train.shape[0], 32);

        for _ in 0..epochs {
            let mut indices = (0..x_train.shape[0]).collect::<Vec<usize>>();
            let mut rand = Rand::new(self.seed);
            rand.shuffle(&mut indices[..]);

            for training_indices in &indices[0..batch_size] {
                let x_batch = x_train.get_row(*training_indices);
                let y_batch = y_train.get_row(*training_indices);
                let loss = self.step(&x_batch, &y_batch);
                if verbose { println!("Loss: {}", loss) };
            }
            
        }
    }


    /// Train the network and return the loss
    pub fn step(&mut self, x_batch: &Tensor, y_batch: &Tensor) -> f64 {
        // Train our network on a given batch of x_batch and y_batch.
        // Size of the batch = # rows of x_batch = # rows of y_batch
        // We first need to run forward to get all layer activations.
        // Then we can run layer.backward going from last to first layer.
        // After we have called backward for all layers, all Dense layers have already made one gradient step.

        // Get the layer activations
        
        let mut layer_activations = self.forward_propagation(x_batch, true);
        layer_activations.insert(0, x_batch.clone());

        // Compute the loss and the initial gradient
        let loss = (&layer_activations.last().unwrap().get_transpose() - &y_batch).map(|x| x*x).data.iter().sum::<f64>();
        
        let mut loss_grad = &layer_activations.last().unwrap().get_transpose() - &y_batch;

        // Propagate gradients through the network
        // Reverse propogation as this is backprop
        for (i, layer) in self.layers.iter_mut().skip(1).rev().enumerate() {
            let i = layer_activations.len() - 2 - i;
            loss_grad = layer.backward(&layer_activations[i], loss_grad);
        }
        
        loss
    }

    pub fn predict(&mut self, input: &Vec<f64>) -> Tensor {
        let tensor_input = Tensor::new(input.to_vec(), vec![1, input.to_vec().len()]);

        // The output of the network is the last layer output
        match self.forward_propagation(&tensor_input, false).last() {
            Some(x) => x.clone(),
            None => panic!("No prediction."),
        }
    }
}
