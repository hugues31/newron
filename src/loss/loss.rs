use crate::tensor::Tensor;

pub trait Loss {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor;
    fn compute_loss_grad(&self);
}
