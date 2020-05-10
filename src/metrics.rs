// TODO: Clean ConfusionMatrix conditions
// TODO: Compute confusion matrix for multiclass

pub enum Metrics {
    Accuracy
}


struct ConfusionMatrix {
    data: Vec<Vec<f64>>
}


/// Compute confusion matrix to evaluate the accuracy of a classification.
/// >>> confusionMatrix(y_true, y_pred)
impl ConfusionMatrix {
    fn new(y_true: Vec<f64>, y_pred: Vec<f64>) -> ConfusionMatrix {
        let mut t_n: f64 = 0.0;
        let mut f_p: f64 = 0.0;
        let mut f_n: f64 = 0.0;
        let mut t_p: f64 = 0.0;

        // TODO: assert y_true et y_pred are the same length
        for (idx, _) in y_true.iter().enumerate() {
            if y_true[idx] == y_pred[idx] && y_true[idx] == 1.0 {
                t_p += 1.0;
            } else if y_true[idx] == y_pred[idx] && y_true[idx] == 0.0 {
                t_n += 1.0;
            } else if y_true[idx] == 0.0 {
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
        ConfusionMatrix { data }
    }

    // fn get_accuracy(&self) -> f64 {
    //      let mut correct_preds = 0;
    //     for index in 0..predictions_categories.len() {
    //         if predictions_categories[index] == true_values_categories[index] {
    //             correct_preds += 1;
    //         }
    //     }
    //     let accuracy = correct_preds as f64 / predictions_categories.len() as f64 * 100.0;
    // }
}
