#[cfg(test)]
mod softmax_tests {
    use newron::layers::layer::Layer;
    use newron::layers::tanh::Tanh;
    use newron::tensor::Tensor;
    use newron::utils;

    #[test]
    fn test_tanh_forward() {
        let layer = Tanh{};

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![0.0, 1.6, 1.0, 3.1], vec![1, 4]);

        let forward = layer.forward(&input);

        let result = Tensor::new(vec![ 0.0, 0.9, 0.8, 1.0], vec![1, 4]);

        assert_eq!(utils::round_vector(forward.data, 1), result.data);
    }

    #[test]
    fn test_tanh_backward() {
        let mut layer = Tanh{};

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![0.0, 1.6, 1.0, 3.1], vec![1, 4]);

        let backward = layer.backward(&input, &layer.forward(&input));
        
        // ~ [ 0.  0.91715234  0.78071444  0.99627208] dot [0.0, 0.9, 0.8, 1.0]
        let result = Tensor::new(vec![0.0, 0.1, 0.3, 0.0], vec![1, 4]);

        assert_eq!(utils::round_vector(backward.data, 1), result.data);
    }
}
