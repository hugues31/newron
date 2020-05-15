#[cfg(test)]
mod metrics_tests {
    use newron::tensor::Tensor;
    use newron::metrics::*;
    use newron::utils;

    #[test]
    fn test_accuracy() {

        let predictions = Tensor::new(vec![0.4, 0.6,
                                           0.1, 0.9,
                                           0.3, 0.7,
                                           1.0, 0.0], vec![4, 2]);

        let true_values = Tensor::new(vec![0.4, 0.6,
                                           0.1, 0.9,
                                           0.7, 0.3,
                                           1.0, 0.0], vec![4, 2]);

        let metric = accuracy::Accuracy{
            y_true: true_values.clone(), 
            y_pred: predictions.clone()
        };

        let acc_score = metric.compute();

        let result = 75.0;
        assert_eq!(utils::round_f64(acc_score, 1), result);
    }

    #[test]
    fn test_cm() {

        let true_values = Tensor::new(vec![0.4, 0.6,
                                           0.1, 0.9,
                                           0.7, 0.3,
                                           1.0, 0.0], vec![4, 2]);

        let predictions = Tensor::new(vec![0.4, 0.6,
                                           0.1, 0.9,
                                           0.3, 0.7,
                                           1.0, 0.0], vec![4, 2]);

        let cm = confusion_matrix::ConfusionMatrix::new(true_values.clone(),
                                                        predictions.clone());

        let result = vec![vec![1, 1],
                          vec![0, 2]];

        assert_eq!(cm.data, result);
    }
}
