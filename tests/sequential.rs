#[cfg(test)]
mod sequential_tests {
    use newron::sequential::Sequential;
    use newron::layer::Layer;
    
    #[test]
    fn test_sequential_stacking() {
        let mut model = Sequential::new();

        let layer_1 = Layer::new(Activation::relu(), 5, 0.0);
        let layer_2 = Layer::new(Activation::relu(), 7, 0.0);

        model.add(layer_1);
        model.add(layer_2);

        assert_eq!(model.layers.len(), 2);
    }
}
