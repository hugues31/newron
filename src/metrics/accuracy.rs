// use crate::metrics::metrics::Metri
use std::fmt;
use crate::utils;
use crate::tensor::Tensor;
//
// input    ->   { y_true <vec>, y_pred <vec> }
// output   ->   { f64 }


pub struct Accuracy{
    y_true: Tensor,
    y_pred: Tensor,
}

impl Accuracy {

    pub fn compute(&self) -> f64 {
        // TODO: refactor (accuracy is not the same for classification or regression)
        let predictions_categories = utils::one_hot_encoded_tensor_to_indices(&self.y_pred);
        let true_values_categories = utils::one_hot_encoded_tensor_to_indices(&self.y_true);
        let mut correct_preds = 0;

        for index in 0..predictions_categories.len() {
            if predictions_categories[index] == true_values_categories[index] {
                correct_preds += 1;
            }
        }
        let accuracy = correct_preds as f64 / predictions_categories.len() as f64 * 100.0;

        accuracy 
    }
}

impl fmt::Display for Accuracy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.y_true, self.y_pred)
    }
}
