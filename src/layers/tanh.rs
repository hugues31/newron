use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;
use crate::layers::layer::LayerInfo;

pub struct TanH {
    input: Tensor
}

impl Layer for TanH {
    fn get_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: format!("tanH"),
            output_shape: vec![],
            trainable_param: 0,
            non_trainable_param: 0,
        }
    }

    fn forward(&mut self, input: Tensor, _training: bool) -> Tensor {
        self.input = input;
        self.input.map(|x| TanH::tanh(x))
    }

    fn backward(&mut self, gradient: &Tensor) -> Tensor {
        let tanh_grad = self.input.map(|x| TanH::tanh_prime(x));
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

impl TanH {
    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    fn tanh_prime(x: f64) -> f64 {
        1.0 - Self::tanh(x).powi(2)
    }

    pub fn new() -> TanH {
        TanH {
            input: Tensor::new(vec![], vec![])
        }
    }
}