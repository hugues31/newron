use newron::activation::Activation;
use newron::dataset::Dataset;
use newron::layer::Layer;
use newron::sequential::Sequential;

fn main() {
    // Let's create a toy dataset
    let dataset = Dataset::from_raw_data(
            vec![
                //   X_0, X_1, X_2, Y
                vec![1.0, 0.0, 1.0, 1.0],
                vec![0.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![1.0, 1.0, 1.0, 0.0],
            ]
    ).unwrap();

    let mut model = Sequential::new();
    
    let hidden_layer_1 = Layer::new(Activation::relu(), 5, 0.0);
    let hidden_layer_2 = Layer::new(Activation::tanh(), 8, 0.2);
    let output_layer = Layer::new(Activation::relu(), 1, 0.0);

    model.add(hidden_layer_1);
    model.add(hidden_layer_2);
    model.add(output_layer);

    model.fit(&dataset, 2_000, true);

    let features_to_predict = vec![0.0, 0.0, 1.0];
    let prediction = model.predict(&features_to_predict);

    println!("Prediction for {:?} : {}", &features_to_predict, &prediction);
}
