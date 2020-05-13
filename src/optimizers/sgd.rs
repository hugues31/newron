use crate::{layers::layer::Layer, optimizers::optimizer::OptimizerStep, tensor::Tensor};

pub struct SGD {
    /// Learning Rate
    lr: f64
}

impl SGD {
    pub fn new(lr: f64) -> Self { Self { lr } }
}

impl OptimizerStep for SGD {
    fn step(&self, layers: &mut [Box<dyn Layer>]) {
        for layer in layers.iter_mut() {
            for param in layer.get_params_list() {
                let grad = layer.get_grad(&param).clone();
                
                let param_to_update = &mut *layer.get_param(&param);
     
                *param_to_update -= self.lr * grad;
            }
        }
    }
}
