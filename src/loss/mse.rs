use crate::{tensor::Tensor, loss::loss::Loss};
pub struct MSE {}

impl Loss for MSE {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f64 {
        let loss = (y_pred - y_true).map(|x| x*x).get_sum(0).data[0];
        loss
    }

    fn compute_loss_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        let loss: Tensor = 2.0 * (y_pred - y_true);
        loss
    }
}