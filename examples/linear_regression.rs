use newron::dataset::{RowType, Dataset, ColumnType};
use newron::layers::LayerEnum::*;
use newron::sequential::Sequential;
use newron::loss::{mse::MSE};
use newron::metrics::Metric;
use newron::optimizers::sgd::SGD;

fn main() {
    // Sample the function f(x) = 0.2x + 2
    // with 20 points (X, Y) starting with X=-50 to X=50
    let mut data = Vec::new();
    for x in -10..10 {
        let x = x as f64;
        let y = 0.2 * x + 2.0;
        data.push(vec![x, y]);
    }

    let mut dataset = Dataset::from_raw_data(data).unwrap();

    // We set 60% of rows for training and 40% for validation
    dataset.split_train_test(0.6, true);
    panic!("{:?}", dataset.get_tensor(RowType::Train, ColumnType::Feature));

    let mut model = Sequential::new();

    // We only need one neuron (slope + intercept (bias))
    model.add(Dense{input_units: 1, output_units:1});

    model.compile(MSE{},
        SGD::new(0.0002),
        vec![Metric::Accuracy]);

    model.summary();

    // We train the model for 200 epochs
    model.fit(&dataset, 200, true);

    // Display the slope and intercept estimated (=Dense weight/bias)
    println!("\n\nEstimated equation :\n{:?}", model.layers[0]);

    // Interpolation (the model did not see any value for X=4.6)
    let value_to_predict = 4.6;
    let prediction = model.predict(&vec![value_to_predict]).get_value(0, 0);
    let true_value = 0.2 * value_to_predict + 2.0;
    println!(
        "Prediction for X={} -> Y={:.4} (true value={:.4})",
        &value_to_predict, &prediction, &true_value
    );
}
