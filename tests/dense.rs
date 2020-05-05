#[cfg(test)]
mod softmax_tests {
    use newron::layers::layer::Layer;
    use newron::layers::dense::Dense;
    use newron::tensor::Tensor;
    use newron::utils;

    #[test]
    fn test_dense_forward() {
        let input_units = 3;
        let output_units = 4;
        let layer = Dense::new(input_units, output_units);

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![1.0, 1.0, 1.0,
                                     2.0, 2.0, 2.0], vec![2, 3]);

        let forward = layer.forward(&input);

        panic!("{:?}", forward);

        let result = Tensor::new(vec![0.0, 3.0, 0.0, 5.0, 0.0, 2.0], vec![1, 6]);

        assert_eq!(utils::round_vector(forward.data, 3), result.data);
    }

    #[test]
    fn test_relu_backward() {
        let input_units = 3;
        let output_units = 4;
        let mut layer = Dense::new(input_units, output_units);

        // Test 1 dimension (when batch = 1 sample for example)
        let input = Tensor::new(vec![1.0, 1.0, 1.0,
            2.0, 2.0, 2.0], vec![2, 3]);

        let backward = layer.backward(&input, &layer.forward(&input));

        panic!("{:?}", backward);
        
        // [0.0, 3.0, 0.0, 5.0, 0.0, 2.0] dot [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        let result = Tensor::new(vec![0.0, 3.0, 0.0, 5.0, 0.0, 2.0], vec![1, 6]);

        assert_eq!(utils::round_vector(backward.data, 3), result.data);
    }
}
