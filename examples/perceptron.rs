use newron::sequential::Sequential;
use newron::layer::Layer;
use newron::tensor::Tensor;

fn main() {
    let x_train = Tensor::new(vec![
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0],
            vec![4,3]);

    let y_train = Tensor::new(vec![
            1.0,
            1.0,
            0.0,
            0.0],
            vec![4,1]);

    let mut model = Sequential::new();
    
    let layer_1 = Layer::new("relu".to_string(), 5, 0.0);
    let layer_2 = Layer::new("relu".to_string(), 6, 0.0);
    let layer_3 = Layer::new("relu".to_string(), 1, 0.0);

    model.add(layer_1);
    model.add(layer_2);
    model.add(layer_3);

    model.fit(x_train, y_train, 1_000, true);
}
