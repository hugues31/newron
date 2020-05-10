
struct ConfusionMatrix {
    data: Vec<Vec<f64>>
}


/// Compute confusion matrix to evaluate the accuracy of a classification.
/// >>> confusionMatrix(y_true, y_pred)
impl ConfusionMatrix {
    fn new(y_true, y_pred) -> ConfusionMatrix {
        let mut t_n = 0;
        let mut f_p = 0;
        let mut f_n = 0;
        let mut t_p = 0;

        // TODO: assert y_true et y_pred are the same length
        for (idx, value) in y_true.iter().enumerate() {
            if y_true[idx] == y_pred[idx] && y_true == 1 {
                t_p += 1;
            } else if y_true[idx] == y_pred[idxx] && y_true == 0 {
                t_n += 1;
            } else if y_pred[idx] == 1 {
                f_p += 1;
            } else {
                f_n += 1;
            }
        }
        (t_n, f_p, f_n, t_p)
    }


    fn get_accuracy(&self) -> f64 {
         let mut correct_preds = 0;
        for index in 0..predictions_categories.len() {
            if predictions_categories[index] == true_values_categories[index] {
                correct_preds += 1;
            }
        }
        let accuracy = correct_preds as f64 / predictions_categories.len() as f64 * 100.0;
    }
}
