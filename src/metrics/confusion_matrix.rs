use crate::utils;
use crate::tensor::Tensor;

pub struct ConfusionMatrix {
    pub data: Vec<Vec<usize>>,
}

/// Compute confusion matrix to evaluate the accuracy of a classification.
///
/// Examples
/// --------
/// >>> metrics::ConfusionMatrix::new(y_true, y_pred)
/// >>> metrics::ConfusionMatrix.accuracy_score()
/// >>> metrics::ConfusionMatrix.recall_score()
impl ConfusionMatrix {
    
    pub fn new(y_true: Tensor, y_pred: Tensor) -> ConfusionMatrix {
        assert_eq!(y_true.shape, y_pred.shape);

        // TODO: warning, does it really get CM shape???
        let cm_shape = vec![y_true.shape[1], y_true.shape[1]];
        let mut cm = vec![vec![0; cm_shape[0]]; cm_shape[0]];

        let y_pred_categories = utils::one_hot_encoded_tensor_to_indices(&y_pred);
        let y_true_categories = utils::one_hot_encoded_tensor_to_indices(&y_true);
        
        for index in 0..y_pred_categories.len() {
            cm[y_true_categories[index]][y_pred_categories[index]] += 1;
        }

        ConfusionMatrix{ data: cm }
    }
    
    /// Compute accuracy score based on confusion matrix
    pub fn accuracy_score(&self) -> f64 {
        let mut correct_classif: usize = 0;

        for i in 0..self.data.len() {
            for j in 0..self.data[i].len() {

                if i == j {
                    correct_classif += self.data[i][j];
                }
            }
        }

        let cm_sum: f64 = self.data.iter()
                                    .map(|v| v.iter().sum::<usize>() as f64)
                                    .sum();

        correct_classif as f64 / cm_sum
    }

    pub fn f1_score(&self) -> () {
        todo!();
    }

    pub fn recall_score(&self) -> () {
        todo!();
    }

    pub fn precision_score(&self) -> () {
        todo!();
    }

}
