use newron::activation::Activation;
use newron::layer::Layer;
use newron::sequential::Sequential;
use newron::tensor::Tensor;

fn main() {
    let x_train = Tensor::new(
        vec![1.0, 0.0, 1.0, 
             0.0, 1.0, 1.0, 
             0.0, 0.0, 1.0, 
             1.0, 1.0, 1.0],
        vec![4, 3],
    );

    let y_train = Tensor::new(vec![1.0, 1.0, 0.0, 0.0], vec![4, 1]);

    let mut model = Sequential::new();

    let hidden_layer_1 = Layer::new(Activation::relu(), 5, 0.0);
    let hidden_layer_2 = Layer::new(Activation::tanh(), 8, 0.07);
    let output_layer = Layer::new(Activation::relu(), 1, 0.0);

    // Setting the seed is optional; By default it is set to zero
    model.set_seed(777);

    model.add(hidden_layer_1);
    model.add(hidden_layer_2);
    model.add(output_layer);

    model.fit(&x_train, &y_train, 2_000, true);

    let prediction = model.predict(&x_train.get_row(2));

    println!("Prediction for 0.0, 0.0, 1.0 : {}", &prediction);
}
