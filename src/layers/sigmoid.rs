use crate::layers::layer::LayerInfo;
use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;
use crate::layers::layer::LayerInfo;

pub struct Sigmoid {
    input: Tensor
}

impl Layer for Sigmoid {
    fn get_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: format!("Sigmoid"),
            output_shape: self.input.shape.to_vec(),
            trainable_param: 0,
            non_trainable_param: 0,

        }
    }

    fn forward(&mut self, input: Tensor, _training: bool) -> Tensor {
        self.input = input;
        self.input.map(|x| Sigmoid::sigmoid(x))
    }

    fn backward(&mut self, gradient: &Tensor) -> Tensor {
        let tanh_grad = self.input.map(|x| Sigmoid::sigmoid_prime(x));
        gradient.mult_el(&tanh_grad)
    }

    fn get_params_list(&self) -> Vec<LearnableParams> {
        vec![]
    }
    
    fn get_grad(&self, _param: &LearnableParams) -> &Tensor {
        panic!("Layer does not have learnable parameters.")
    }

    fn get_param(&mut self, _param: &LearnableParams) -> &mut Tensor {
        panic!("Layer does not have learnable parameters.")

    }
}

impl Sigmoid {
    pub fn sigmoid(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }

    pub fn sigmoid_prime(x: f64) -> f64 {
        Self::sigmoid(x) * (1. - Self::sigmoid(x))
    }

    pub fn new() -> Sigmoid {
        Sigmoid {
            input: Tensor::new(vec![], vec![])
        }
    }

}
