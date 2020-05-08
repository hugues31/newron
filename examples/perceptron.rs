use newron::dataset::Dataset;
use newron::layers::LayerEnum::*;
use newron::sequential::Sequential;
use newron::loss::{mse::MSE};
use newron::metrics::Metrics;
use newron::optimizers::sgd::SGD;

fn main() {
    // Let's create a toy dataset
    let dataset = Dataset::from_raw_data(vec![
        //   X_0, X_1, X_2, Y
        vec![1.0, 0.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0, 0.0],
    ])
    .unwrap();

    let mut model = Sequential::new();
    
    model.set_seed(42);

    model.add(Dense{input_units:3, output_units:4});
    model.add(TanH);

    model.add(Dense{input_units:4, output_units:1});

    model.compile(MSE{},
        SGD::new(0.02),
        vec![Metrics::Accuracy]);

    model.summary();

    model.fit(&dataset, 500, true);

    let features_to_predict = vec![1.0, 0.0, 1.0];
    let prediction = model.predict(&features_to_predict);

    println!(
        "Prediction for {:?} : {}",
        &features_to_predict, &prediction
    );
}
