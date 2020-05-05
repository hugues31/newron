#[cfg(test)]
mod softmax_tests {
    use newron::layers::layer::Layer;
    use newron::layers::relu::ReLU;
    use newron::tensor::Tensor;
    use newron::utils;

    #[test]
    fn test_relu_forward() {
        let layer = ReLU{};

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![-1.0, 3.0, -2.5, 5.0, -4.0, 2.0], vec![1, 6]);

        let forward = layer.forward(&input);

        let result = Tensor::new(vec![0.0, 3.0, 0.0, 5.0, 0.0, 2.0], vec![1, 6]);

        assert_eq!(utils::round_vector(forward.data, 1), result.data);
    }

    #[test]
    fn test_relu_backward() {
        let mut layer = ReLU{};

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![-1.0, 3.0, -2.5, 5.0, -4.0, 2.0], vec![1, 6]);

        let backward = layer.backward(&input, layer.forward(&input));
        
        // [0.0, 3.0, 0.0, 5.0, 0.0, 2.0] ⊙ [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        let result = Tensor::new(vec![0.0, 3.0, 0.0, 5.0, 0.0, 2.0], vec![1, 6]);

        assert_eq!(utils::round_vector(backward.data, 1), result.data);
    }
}
