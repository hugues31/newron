use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use crate::layers::layer::LearnableParams;
use crate::layers::layer::LayerInfo;

pub struct ReLU {
    input: Tensor
}

impl Layer for ReLU {
    fn get_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: format!("ReLU"),
            output_shape: self.input.shape.to_vec(),
            trainable_param: 0,
            non_trainable_param: 0,
        }
    }

    fn forward(&mut self, input: Tensor, _training: bool) -> Tensor {
        self.input = input;
        self.input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, gradient: &Tensor) -> Tensor {
        let relu_grad = self.input.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        gradient.mult_el(&relu_grad)
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

impl ReLU {
    pub(crate) fn new() -> ReLU {
        ReLU {
            input: Tensor::new(vec![], vec![])
        }
    }
}
