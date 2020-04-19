use newron::sequential::Sequential;
use newron::layer::Layer;
use newron::tensor::Tensor;
use newron::activation::Activation;

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
    
    let hidden_layer_1 = Layer::new(Activation::relu(), 5, 0.3);
    let hidden_layer_2 = Layer::new(Activation::tanh(), 8, 0.0);
    let output_layer = Layer::new(Activation::relu(), 1, 0.0);

    model.add(hidden_layer_1);
    model.add(hidden_layer_2);
    model.add(output_layer);

    model.fit(&x_train, &y_train, 2_000, true);

    let prediction = model.predict(&x_train.get_row(2));

    println!("Prediction for 0.0, 0.0, 1.0 : {}", &prediction);
}
