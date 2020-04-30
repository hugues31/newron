use crate::{tensor::Tensor, loss::loss::Loss};
pub struct CategoricalEntropy {}

impl Loss for CategoricalEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        unimplemented!()
    }

    fn compute_loss_grad(&self) {
        unimplemented!()
    }
}