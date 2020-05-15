use crate::tensor::Tensor;
use crate::utils;

pub struct ConfusionMatrix {
    pub data: Vec<Vec<usize>>,
}

/// Compute confusion matrix to evaluate the accuracy of a classification.
///
/// Examples
/// --------
/// >>> metrics::ConfusionMatrix::new(y_true, y_pred)
/// >>> metrics::ConfusionMatrix.accuracy_score()
/// >>> metrics::ConfusionMatrix.recall_score(1)
/// >>> metrics::ConfusionMatrix.precision_score(1)
/// >>> metrics::ConfusionMatrix.f1_score_score(1)
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

        ConfusionMatrix { data: cm }
    }

    /// Compute accuracy score based on confusion matrix
    pub fn accuracy_score(&self) -> f64 {
        let correct_classif: f64 =
            (0..self.data.len()).map(|v| self.data[v][v]).sum::<usize>() as f64;

        let cm_sum: f64 = self
            .data
            .iter()
            .map(|v| v.iter().sum::<usize>() as f64)
            .sum();

        correct_classif as f64 / cm_sum
    }

    /// The recall for input class is the number of
    /// correctly predicted  input class out of the number of actual input class
    pub fn recall_score(&self, class: usize) -> f64 {
        let correct_class: usize = self.data[class][class];
        let all_predicted_class = (0..self.data.len())
            .map(|v| self.data[v][class])
            .sum::<usize>() as f64;

        correct_class as f64 / all_predicted_class
    }

    /// The precision for the input class is the number of
    /// correctly predicted input class out of all predicted input class
    pub fn precision_score(&self, class: usize) -> f64 {
        let correct_class: usize = self.data[class][class];
        let actual_class = (0..self.data.len())
            .map(|v| self.data[class][v])
            .sum::<usize>() as f64;

        correct_class as f64 / actual_class
    }

    /// Harmonic mean of the precision and recall
    pub fn f1_score(&self, class: usize) -> f64 {
        (2.0 * self.recall_score(class) * self.precision_score(class))
            / (self.recall_score(class) + self.precision_score(class))
    }

}
