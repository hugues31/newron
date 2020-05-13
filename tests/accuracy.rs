#[cfg(test)]
mod metrics_tests {
    use newron::tensor::Tensor;
    use newron::metrics::MetricEnum;
    use newron::metrics::*;
    use newron::utils;

    #[test]
    fn test_accuracy() {

        let predictions = Tensor::new(vec![1.0,
                                           0.0,
                                           0.0,
                                           1.0], vec![4, 1]);

        let true_values = Tensor::new(vec![1.0,
                                           0.0,
                                           1.0,
                                           1.0], vec![4, 1]);

        let metric = accuracy::Accuracy{
            y_true: true_values.clone(), 
            y_pred: predictions.clone()
        };

        let acc_score = metric.compute();

        let result = 75.0;
        println!("{}", predictions);
        println!("{}", true_values);
        println!("{}", acc_score);
        assert_eq!(utils::round_f64(acc_score, 2), result);
    }
}
