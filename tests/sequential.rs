#[cfg(test)]
mod sequential_tests {
    use newron::sequential::Sequential;
    use newron::layers::LayerEnum::*;
    use newron::loss::{mse::MSE};
    use newron::optimizers::sgd::SGD;
    use newron::metrics::Metrics;
    
    #[test]
    fn test_sequential_stacking() {
        let mut model = Sequential::new();

        model.add(Dense {input_units: 20, output_units: 100});
        model.add(ReLU);

        model.compile(MSE{},
            SGD::new(0.002),
            vec![Metrics::Accuracy]);

        assert_eq!(model.layers.len(), 2);
    }
}
