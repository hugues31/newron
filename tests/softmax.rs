#[cfg(test)]
mod softmax_tests {
    use newron::layers::layer::Layer;
    use newron::layers::softmax::Softmax;
    use newron::tensor::Tensor;
    use newron::utils;

    #[test]
    fn test_softmax_forward() {
        let layer = Softmax{};

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![1.0, 3.0, 2.5, 5.0, 4.0, 2.0], vec![1, 6]);
        let forward = layer.forward(&input);

        let result = Tensor::new(vec![0.011, 0.082, 0.05, 0.605, 0.222, 0.03], vec![1, 6]);

        assert_eq!(utils::round_vector(forward.data, 3), result.data);

        // Test 3 dimensions (batch = 3 samples)
        let input = Tensor::new(vec![1.0, 2.0,
                                     2.0, 3.0,
                                     2.0, 2.0], vec![3, 2]);
        let forward = layer.forward(&input);

        let result = Tensor::new(vec![0.269, 0.731,
                                      0.269, 0.731,
                                      0.5, 0.5], vec![3, 2]);

        assert_eq!(utils::round_vector(forward.data, 3), result.data);
    }

    #[test]
    fn test_softmax_backward() {
        let mut layer = Softmax{};

        // Test 3 dimensions (batch = 3 samples)
        let input = Tensor::new(vec![1.0, 2.0, 3.0,
                                     2.0, 4.0, 2.0,
                                     -3.0, 2.0, 0.0,
                                     2.0, 4.0, 2.0], vec![4, 3]);

        let previous_grad = Tensor::new(vec![1.0, 2.0, 0.0], vec![1, 3]);

        let backward = layer.backward(&input, &previous_grad);

        let result = Tensor::new(vec![0.038, 0.348, -0.385,
                                                   -0.072, 0.251, -0.179,
                                                   -0.004, 0.213, -0.208,
                                                   -0.072, 0.251, -0.179], vec![4, 3]);

        assert_eq!(utils::round_vector(backward.data, 3), result.data);
    }
}
