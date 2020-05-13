use std::fmt;
use crate::utils;
use crate::tensor::Tensor;

// TODO: Clean ConfusionMatrix conditions
// TODO: Compute confusion matrix for multiclass

pub struct ConfusionMatrix {
    pub y_true: Tensor,
    pub y_pred: Tensor,
}


/// Compute confusion matrix to evaluate the accuracy of a classification.
/// >>> confusionMatrix(y_true, y_pred)
impl ConfusionMatrix {
    fn compute(&self) -> Vec<Vec<f64>> {
        let mut t_n: f64 = 0.0;
        let mut f_p: f64 = 0.0;
        let mut f_n: f64 = 0.0;
        let mut t_p: f64 = 0.0;

        // TODO: assert y_true et y_pred are the same length
        for idx in 0..self.y_true.shape[0] {
            if self.y_true[idx] == self.y_pred[idx] && self.y_true[idx] == 1.0 {
                t_p += 1.0;
            } else if self.y_true[idx] == self.y_pred[idx] && self.y_true[idx] == 0.0 {
                t_n += 1.0;
            } else if self.y_true[idx] == 0.0 {
                f_p += 1.0;
            } else {
                f_n += 1.0;
            }
        }

        let mut data: Vec<Vec<f64>> = vec![];

        data[0].push(t_p);
        data[0].push(f_p);

        data[1].push(f_n);
        data[1].push(t_n);

        println!("{:?}", data);
        data
    }

}
