#[cfg(test)]
mod sigmoid_tests {
    use newron::layers::layer::Layer;
    use newron::layers::sigmoid::Sigmoid;
    use newron::tensor::Tensor;
    use newron::utils;

    #[test]
    fn test_sigmoid_forward() {
        let layer = Sigmoid{};

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![ 0.2, 0.4, 0.1], vec![1, 3]);

        let forward = layer.forward(&input);

        let result = Tensor::new(vec![0.5, 0.6, 0.5], vec![1, 3]);

        assert_eq!(utils::round_vector(forward.data, 1), result.data);
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut layer = Sigmoid{};

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![0.2, 0.4, 0.1], vec![1, 3]);

        let backward = layer.backward(&input, layer.forward(&input));
        
        // ~ [0.3, 0.2, 0.3] ⊙ [0.5, 0.6, 0.5]
        let result = Tensor::new(vec![0.1, 0.1, 0.1], vec![1, 3]);

        assert_eq!(utils::round_vector(backward.data, 1), result.data);
    }
}
