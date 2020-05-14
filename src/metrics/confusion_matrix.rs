use crate::utils;
use crate::tensor::Tensor;

// TODO: Clean ConfusionMatrix conditions
// TODO: Compute confusion matrix for multiclass

pub struct ConfusionMatrix {
    pub y_true: Tensor,
    pub y_pred: Tensor,
}


/// Compute confusion matrix to evaluate the accuracy of a classification.
///
/// >>> confusionMatrix(y_true, y_pred)
impl ConfusionMatrix {
    
    pub fn compute(&self) -> Tensor {
        // TODO: refactor (accuracy is not the same for classification or regression)
        assert_eq!(&self.y_true.shape, &self.y_pred.shape);
        let cm_shape = vec![self.y_true.shape[1], self.y_true.shape[1]];

        let mut flat_cm = vec![0.0; cm_shape[0].pow(2)];

        let predictions_categories = utils::one_hot_encoded_tensor_to_indices(&self.y_pred);
        let true_values_categories = utils::one_hot_encoded_tensor_to_indices(&self.y_true);
        
        for index in 0..predictions_categories.len() {
            flat_cm[true_values_categories[index] * cm_shape[0] + predictions_categories[index]] += 1.0;
        }

        let cm = Tensor::new(flat_cm.clone(), cm_shape.clone());

        cm
    }

    pub fn accuracy_score(&self) -> () {
        //TODO
        ()
    }

    pub fn f1_score(&self) -> () {
        //TODO
        ()
    }

    pub fn recall_score(&self) -> () {
        //TODO
        ()
    }

    pub fn precision_score(&self) -> () {
        //TODO
        ()
    }

}
