#[cfg(test)]
mod mse_tests {
    use newron::tensor::Tensor;
    use newron::loss::loss::Loss;
    use newron::loss::mse::MSE;
    use newron::utils;

    #[test]
    fn test_mse_loss() {
        let loss = MSE{};

        // Test 3 dimensions (batch = 3 samples)
        let predictions = Tensor::new(vec![10.2,
                                           17.8,
                                           22.0], vec![3, 1]);

        let true_values = Tensor::new(vec![9.4,
                                           17.5,
                                           23.9,], vec![3, 1]);

        let loss = loss.compute_loss(&true_values, &predictions);

        let result = 1.447;

        assert_eq!(utils::round_f64(loss, 3), result);
    }

    #[test]
    fn test_mse_loss_grad() {
        let loss = MSE{};

        // Test 3 dimensions (batch = 3 samples)
        let predictions = Tensor::new(vec![10.2,
            17.8,
            22.0], vec![3, 1]);

        let true_values = Tensor::new(vec![9.4,
                    17.5,
                    23.9,], vec![3, 1]);

        let loss_grad = loss.compute_loss_grad(&true_values, &predictions);

        let result = Tensor::new(vec![1.6, 0.6, -3.8], vec![1, 3]);

        assert_eq!(utils::round_vector(loss_grad.data, 3), result.data);
    }
}