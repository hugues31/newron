use std::path::Path;

use newron::activation::Activation;
use newron::dataset::{Dataset,RowType, ColumnType};
use newron::layer::Layer;
use newron::sequential::Sequential;

fn main() {
    let dataset = Dataset::from_csv(Path::new("datasets/winequality-white.csv"), true).unwrap();

    println!("{:?}", dataset);
    println!("{:?}", dataset.get_tensor(RowType::Train, ColumnType::Target));

    let mut model = Sequential::new();

    let hidden_layer_1 = Layer::new(Activation::relu(), 11, 0.0);
    // let hidden_layer_2 = Layer::new(Activation::tanh(), 80, 0.0);
    let output_layer = Layer::new(Activation::relu(), 1, 0.0);

    model.set_seed(0);

    model.add(hidden_layer_1);
    // model.add(hidden_layer_2);
    model.add(output_layer);

    model.fit(&dataset, 2_000, true);
}
