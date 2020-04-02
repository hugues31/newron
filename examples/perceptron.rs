use newron::sequential::Sequential;
use newron::layer::Layer;
use newron::Matrix;


fn main() {
    let x_train = Matrix::from_row_slice(4,3,&[
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0]);

    let y_train = Matrix::from_row_slice(4,1,&[1.0, 1.0, 0.0, 0.0]);

    let mut model = Sequential::new();
    
    let layer_1 = Layer::new("relu".to_string(), 5, 3, 0.0);
    let layer_2 = Layer::new("relu".to_string(), 6, 5, 0.0);
    let layer_3 = Layer::new("relu".to_string(), 1, 6, 0.0);

    model.add(layer_1);
    model.add(layer_2);
    model.add(layer_3);

    model.fit(x_train, y_train, 1_000, true);
}
