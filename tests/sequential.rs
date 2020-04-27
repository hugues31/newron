#[cfg(test)]
mod sequential_tests {
    use newron::sequential::Sequential;
    use newron::layers::{dense::Dense, relu::ReLU};
    use newron::activation::Activation;
    
    #[test]
    fn test_sequential_stacking() {
        let mut model = Sequential::new();

        model.add(ReLU);
        model.add(ReLU);

        assert_eq!(model.layers.len(), 2);
    }
}
