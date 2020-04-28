#[cfg(test)]
mod sequential_tests {
    use newron::sequential::Sequential;
    use newron::layers::{dense::Dense, relu::ReLU};
    
    #[test]
    fn test_sequential_stacking() {
        let mut model = Sequential::new();

        model.add(Dense::new(20, 30));
        model.add(ReLU);

        assert_eq!(model.layers.len(), 2);
    }
}
