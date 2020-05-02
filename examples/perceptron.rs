use newron::dataset::Dataset;
use newron::layers::{dense::Dense, relu::ReLU, softmax::Softmax};
use newron::sequential::Sequential;

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

    model.add(Dense::new(3, 8));
    model.add(ReLU);

    model.add(Dense::new(8, 1));
    model.add(ReLU);

    model.fit(&dataset, 42, true);

    let features_to_predict = vec![1.0, 0.0, 1.0];
    let prediction = model.predict(&features_to_predict);

    println!(
        "Prediction for {:?} : {}",
        &features_to_predict, &prediction
    );
}
