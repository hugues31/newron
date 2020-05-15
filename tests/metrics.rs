#[cfg(test)]
mod metrics_tests {
    use newron::metrics::*;
    use newron::tensor::Tensor;
    use newron::utils;

    fn setup() -> (Tensor, Tensor) {
        let predictions: Tensor = Tensor::new(vec![0.4, 0.6, 
                                                   0.1, 0.9, 
                                                   0.3, 0.7, 
                                                   1.0, 0.0], vec![4, 2]);

        let true_values: Tensor = Tensor::new(vec![0.4, 0.6, 
                                                   0.1, 0.9, 
                                                   0.7, 0.3, 
                                                   1.0, 0.0], vec![4, 2]);
        (true_values, predictions)
    }

    #[test]
    fn test_cm() {
        let (y_true, y_pred) = setup();

        let cm = confusion_matrix::ConfusionMatrix::new(y_true.clone(), 
                                                        y_pred.clone());

        let result = vec![vec![1, 1], vec![0, 2]];
        assert_eq!(cm.data, result);
    }


    #[test]
    fn test_accuracy() {
        let (y_true, y_pred) = setup();

        let cm = confusion_matrix::ConfusionMatrix::new(y_true.clone(), 
                                                        y_pred.clone());

        let acc_score = cm.accuracy_score();
        let result = 0.75;
        assert_eq!(utils::round_f64(acc_score, 2), result);
    }

    #[test]
    fn test_recall() {
        let (y_true, y_pred) = setup();

        let cm = confusion_matrix::ConfusionMatrix::new(y_true.clone(), 
                                                        y_pred.clone());

        let rec_score = cm.recall_score(1);
        let result = 0.7;
        assert_eq!(utils::round_f64(rec_score, 1), result);
    }

    #[test]
    fn test_precision() {
        let (y_true, y_pred) = setup();

        let cm = confusion_matrix::ConfusionMatrix::new(y_true.clone(), 
                                                        y_pred.clone());

        let pre_score = cm.precision_score(1);
        let result = 1.0;
        assert_eq!(utils::round_f64(pre_score, 1), result);
    }

    #[test]
    fn test_f1() {
        let (y_true, y_pred) = setup();

        let cm = confusion_matrix::ConfusionMatrix::new(y_true.clone(), 
                                                        y_pred.clone());

        let f1_score = cm.f1_score(1);
        let result = 0.8;
        assert_eq!(utils::round_f64(f1_score, 1), result);
    }
}
