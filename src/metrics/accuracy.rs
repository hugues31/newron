use std::fmt;
use crate::utils;
use crate::tensor::Tensor;


pub struct Accuracy{
    pub y_true: Tensor,
    pub y_pred: Tensor,
}

impl Accuracy {

    pub fn compute(&self) -> f64 {
        // TODO: refactor (accuracy is not the same for classification or regression)
        // TODO: one_hot_encoded_tensor_to_indices renvoie un vecteru de 0 
        let predictions_categories = utils::one_hot_encoded_tensor_to_indices(&self.y_pred);
        let true_values_categories = utils::one_hot_encoded_tensor_to_indices(&self.y_true);
        let mut correct_preds = 0;
        
        println!("Y preds: {:?}", predictions_categories);
        println!("y true: {:?}", true_values_categories);

        for index in 0..predictions_categories.len() {
            if &self.y_pred.get_value(index, 0) == &self.y_true.get_value(index, 0) {
                correct_preds += 1;
            }
        }

        let accuracy = correct_preds as f64 / predictions_categories.len() as f64 * 100.0;

        println!("correct preds: {}", correct_preds);

        accuracy 
    }
}

impl fmt::Display for Accuracy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: print self.accuracy_score instead
        write!(f, "({}, {})", self.y_true, self.y_pred)
    }
}
