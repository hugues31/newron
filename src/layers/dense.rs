use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;

pub struct Dense {
    input: Tensor,
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor
}

impl Dense {
    pub fn new(input_units: usize, output_units: usize, seed: u32) -> Dense {
        // initialize with random values following special normal distribution
        // allowing theoritical faster convergence (Xavier Initialization)
        Dense {
            input: Tensor::new(vec![], vec![]),
            weights: Tensor::random_normal(vec![input_units, output_units], 0.0, 2.0 / (input_units + output_units) as f64, seed),
            biases: Tensor::one(vec![1, output_units]),
            weights_grad: Tensor::new(vec![], vec![]),
            biases_grad: Tensor::new(vec![], vec![])
        }
    }
}

impl Layer for Dense {
    fn get_info(&self) -> String {
        format!("Weights {}\nBiases {}", self.weights, self.biases)
    }

    fn forward(&mut self, input: Tensor) -> Tensor {
        // Perform an affine transformation:
        // f(x) = <W*x> + b
        
        // input shape: [batch, input_units]
        // output shape: [batch, output units]
        self.input = input;
        // panic!("input: {:?}", &self.input * &self.weights);

        &self.input * &self.weights + &self.biases
        // &self.input.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, gradient: &Tensor) -> Tensor {
        // compute d f / d x = d f / d dense * d dense / d x
        // where d dense/ d x = weights transposed
        // panic!("input.T {:?}  grad {:?}", &self.input.get_transpose().shape, gradient.shape);
        self.weights_grad = &self.input.get_transpose() * gradient;
        self.biases_grad = gradient.get_sum(0);

        assert_eq!(self.weights_grad.shape, self.weights.shape, "Wrong shape for weight gradients.");
        assert_eq!(self.biases_grad.shape, self.biases.shape, "Wrong shape for biases gradients.");

        let grad_input = gradient * &self.weights.get_transpose();
        grad_input
    }

    fn get_params_list(&self) -> Vec<LearnableParams> {
        vec![LearnableParams::Weights, LearnableParams::Biases]
    }

    fn get_grad(&self, param: &LearnableParams) -> &Tensor {
        match param {
            LearnableParams::Weights => {
                &self.weights_grad
            }
            LearnableParams::Biases => {
                &self.biases_grad
            }
        }
    }

    fn get_param(&mut self, param: &LearnableParams) -> &mut Tensor {
        match param {
            LearnableParams::Weights => {
                &mut self.weights
            }
            LearnableParams::Biases => {
                &mut self.biases
            }
        }
    }

}