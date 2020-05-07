/// The Sequential model is a linear stack of layers.
use std::cmp;

use crate::layers::layer::Layer;
use crate::layers::*;
use crate::layers::LayerEnum;
use crate::tensor::Tensor;
use crate::dataset::{Dataset, RowType, ColumnType};
use crate::{loss::loss::Loss, random::Rand, optimizers::optimizer::OptimizerStep, optimizers::sgd::SGD, metrics::Metrics};
use crate::loss::categorical_entropy::CategoricalEntropy;
use crate::utils;

pub struct Sequential {
    pub layers_enum: Vec<LayerEnum>,
    pub layers: Vec<Box<dyn Layer>>,
    loss: Box<dyn Loss>,
    optim: Box<dyn OptimizerStep>,
    metrics: Vec<Metrics>,
    seed: u32,
}

impl Sequential {
    /// Create a new empty Sequential model.
    pub fn new() -> Sequential {
        Sequential {
            layers_enum: vec![],
            layers: vec![],
            loss: Box::new(CategoricalEntropy{}),
            optim: Box::new(SGD::new(0.02)),
            metrics: vec![],
            seed: 0,
        }
    }

    /// Seed the random number generator
    pub fn set_seed(&mut self, s: u32) {
        self.seed = s;
    }

    /// Add a layer to the model
    pub fn add(&mut self, layer: LayerEnum) {
        self.layers_enum.push(layer);
    }

    /// Get a summary of the model
    pub fn summary(&self) {
        // TODO: add more infos
        println!("Sequential model using {} layers.", self.layers.len());
    }

    pub fn compile<T: 'static + Loss, U: 'static + OptimizerStep>(&mut self, loss: T, optim: U, metrics: Vec<Metrics>) {
        // Set options
        self.loss = Box::new(loss);
        self.optim = Box::new(optim);
        self.metrics = metrics;

        // Build layers
        self.layers.clear();
        for layer in &self.layers_enum {
            self.layers.push(
                match layer {
                    LayerEnum::Dense { input_units, output_units } => {
                        Box::new(dense::Dense::new(*input_units, *output_units, self.seed))
                    }
                    LayerEnum::ReLU => {
                        Box::new(relu::ReLU::new())
                    }
                    LayerEnum::Softmax => {
                        Box::new(softmax::Softmax::new())
                    }
                    LayerEnum::TanH => {
                        Box::new(tanh::TanH::new())
                    }
                    LayerEnum::Sigmoid => {
                        Box::new(sigmoid::Sigmoid::new())
                    }
                }
            );
        }
  
    }

    // Return the list of layer outputs given an input
    fn forward_propagation(&mut self, input: Tensor, train: bool) -> Vec<Tensor> {
        // Compute activations of all network layers by applying them sequentially.
        // Return a list of activations for each layer.
        // Note : Tensor may include one or several rows

        let mut activations: Vec<Tensor> = Vec::new();
        activations.push(input);
        
        // Next propagations with the last propagated values
        for layer in self.layers.iter_mut() {
            let activation = layer.forward(activations.last().unwrap().clone());
            activations.push(activation);
        }

        assert_eq!(activations.len(), self.layers.len() + 1);
        activations
    }

    /// Return a vector containing all batch
    /// if `shuffle` is set to true, batches are randomized
    fn get_batches(&mut self, dataset: &Dataset, batch_size: usize, shuffle: bool) -> Vec<(Tensor, Tensor)> {
        let x_train = dataset.get_tensor(RowType::Train, ColumnType::Feature); 
        let y_train = dataset.get_tensor(RowType::Train, ColumnType::Target);
        
        let mut indices = (0..x_train.shape[0]).collect::<Vec<usize>>();

        if shuffle {
            let mut rand = Rand::new(self.seed);
            rand.shuffle(&mut indices[..]);
            self.seed += 1;
        }

        let mut result = Vec::new();

        for batch_index in (0..x_train.shape[0]).rev().skip(batch_size - 1).step_by(batch_size).rev() {
            let batch_indices: &[usize] = &indices[batch_index..batch_index + batch_size];

            let x_batch = x_train.get_rows(batch_indices);

            let y_batch = y_train.get_rows(batch_indices);
            result.push((x_batch, y_batch));
        }
    
        result
    }

    /// Use this function to train the model on x_train with target y_train.
    /// Set `verbose` to true to see debugging and training information.
    pub fn fit(&mut self, dataset: &Dataset, epochs: u32, verbose: bool) {

        // TODO: Check model architecture (input_unit == x_train.len(),
        // output_unit_l == input_unit_l+1, output_unit_l_n = y_train.len()) and display message here
        
        // auto batch size : TODO improve it
        let batch_size = cmp::min(dataset.get_row_count(), 16);

        for epoch in 0..epochs {
            let batches = self.get_batches(dataset, batch_size, true);

            for batch in batches {
                let x_batch = batch.0;
                let y_batch = batch.1;
                
                self.train(x_batch, &y_batch);
            }

            if verbose {
                println!("Epoch: {}", epoch);

                let train_predictions = self.predict_tensor(dataset.get_tensor(RowType::Train, ColumnType::Feature));
                let train_true_values = &dataset.get_tensor(RowType::Train, ColumnType::Target);
                let train_loss = self.loss.compute_loss(train_true_values, &train_predictions);
                println!("Train error: {}", train_loss);

                if dataset.count_row_type(&RowType::Test) > 0 {
                    let test_predictions = self.predict_tensor(dataset.get_tensor(RowType::Test, ColumnType::Feature));
                    let test_true_values = &dataset.get_tensor(RowType::Test, ColumnType::Target);
                    assert_eq!(test_predictions.shape, test_true_values.shape, "Something wrong happened... o_O");
                    let test_loss = self.loss.compute_loss(test_true_values, &test_predictions);
                    println!("Test error: {}", test_loss);

                    for metric in &self.metrics {
                        match metric {
                            Metrics::Accuracy => {
                                // TODO: refactor (accuracy is not the same for classification or regression)
                                let predictions_categories = utils::one_hot_encoded_tensor_to_indices(&test_predictions);
                                let true_values_categories = utils::one_hot_encoded_tensor_to_indices(&test_true_values);
                                let mut correct_preds = 0;
                                for index in 0..predictions_categories.len() {
                                    if predictions_categories[index] == true_values_categories[index] {
                                        correct_preds += 1;
                                    }
                                }
                                let accuracy = correct_preds as f64 / predictions_categories.len() as f64 * 100.0;

                                println!("Accuracy : {:.2}%", accuracy);
                            }
                        }
                    }
                }
            }
        }
    }


    /// Train the network and return the loss
    pub fn train(&mut self, x_batch: Tensor, y_batch: &Tensor) {
        // Train our network on a given batch of x_batch and y_batch.
        // Size of the batch = # rows of x_batch = # rows of y_batch
        // We first need to run forward to get all layer activations.
        // Then we can run layer.backward going from last to first layer.

        // Get the layer activations
        let x_batch_clone = x_batch.clone();
        let mut layer_activations = self.forward_propagation(x_batch, true);
        layer_activations.insert(0, x_batch_clone);

        let size_batch = y_batch.shape[0];

        // compute loss and average loss gradient
        let loss = self.loss.compute_loss(&y_batch, &layer_activations.last().unwrap());

        // Compute the loss gradient
        let loss_grad = self.loss.compute_loss_grad(&y_batch, &layer_activations.last().unwrap());

        let mut gradient = loss_grad;

        // Propagate gradients through the network layers (backpropagation)
        for layer in self.layers.iter_mut().rev() {
            gradient = layer.backward(&gradient);
        }

        // Update parameters according to the Optimizer specified
        self.optim.step(&mut self.layers);

    }

    pub fn predict(&mut self, input: &Vec<f64>) -> Tensor {
        let tensor_input = Tensor::new(input.to_vec(), vec![1, input.to_vec().len()]);
        self.predict_tensor(tensor_input)
    }

    pub fn predict_tensor(&mut self, input: Tensor) -> Tensor {
        // The output of the network is the last layer output
        match self.forward_propagation(input, false).last() {
            Some(x) => x.clone(),
            None => panic!("No prediction."),
        }
    }
}
