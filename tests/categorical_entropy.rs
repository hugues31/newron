#[cfg(test)]
mod categorical_entropy_tests {
    use newron::tensor::Tensor;
    use newron::loss::loss::Loss;
    use newron::loss::categorical_entropy::CategoricalEntropy;
    use newron::utils;

    #[test]
    fn test_categorical_entropy_loss() {
        let loss = CategoricalEntropy{};

        // Test 3 dimensions (batch = 3 samples)
        let predictions = Tensor::new(vec![0.2, 0.6, 0.2,
                                                         0.8, 0.2, 0.0,
                                                         1.0, 0.0, 0.0,
                                                         0.5, 0.5, 0.0,
                                                         0.0, 0.5, 0.5], vec![5, 3]);

        let true_values = Tensor::new(vec![0.0, 1.0, 0.0,
                                                         1.0, 0.0, 0.0,
                                                         1.0, 0.0, 0.0,
                                                         0.0, 0.0, 1.0,
                                                         0.0, 1.0, 0.0], vec![5, 3]);

        let loss = loss.compute_loss(&true_values, &predictions);

        let result = 0.902;

        assert_eq!(utils::round_f64(loss, 3), result);
    }

    #[test]
    fn test_categorical_entropy_loss_grad() {
        let loss = CategoricalEntropy{};

        // Test 3 dimensions (batch = 3 samples)
        let predictions = Tensor::new(vec![0.2, 0.6, 0.2,
            0.8, 0.2, 0.0,
            1.0, 0.0, 0.0,
            0.5, 0.5, 0.0,
            0.0, 0.5, 0.5], vec![5, 3]);

        let true_values = Tensor::new(vec![0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0], vec![5, 3]);

        let loss = loss.compute_loss_grad(&true_values, &predictions);

        let result = Tensor::new(vec![0.057, -0.115, 0.057,
                                                   -0.1, 0.055, 0.045,
                                                   -0.085, 0.042, 0.042,
                                                    0.077, 0.077, -0.153,
                                                    0.047, -0.123, 0.077], vec![5, 3]);

        assert_eq!(utils::round_vector(loss.data, 3), result.data);
    }
}
