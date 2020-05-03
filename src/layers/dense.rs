use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct Dense {
    weights: Tensor,
    biases: Tensor
}

impl Dense {
    pub fn new(input_units: usize, output_units: usize) -> Dense {
        // initialize with random values following special normal distribution
        // allowing theoritical faster convergence (Xavier Initialization)
        Dense {
            weights: Tensor::random_normal(vec![input_units, output_units], 0.0, 2.0 / (input_units + output_units) as f64,input_units as u32),
            biases: Tensor::one(vec![1, output_units])
        }
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Perform an affine transformation:
        // f(x) = <W*x> + b
        
        // input shape: [batch, input_units]
        // output shape: [batch, output units]

        &input.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, input: &Tensor, grad_output: Tensor) -> Tensor {
        // compute d f / d x = d f / d dense * d dense / d x
        // where d dense/ d x = weights transposed
        let grad_input = &grad_output * &self.weights.get_transpose();
        
        // compute gradient w.r.t. weights and biases
        let grad_weights = &input.get_transpose() * &grad_output;
        let input_rows = input.shape[0] as f64;

        let grad_biases = input_rows * grad_output.get_mean(0);
        assert_eq!(grad_weights.shape, self.weights.shape, "Wrong shape for weight gradients.");
        assert_eq!(grad_biases.shape, self.biases.shape, "Wrong shape for biases gradients.");

        let alpha = 0.2;

        self.weights -= alpha * grad_weights;
        self.biases -= alpha * grad_biases;

        grad_input
    }
}