use std::path::Path;

use newron::activation::Activation;
use newron::dataset::{Dataset,RowType, ColumnType};
use newron::layers::{dense::Dense, relu::ReLU};
use newron::sequential::Sequential;

fn main() {
    let dataset = Dataset::from_csv(Path::new("datasets/winequality-white.csv"), true).unwrap();

    println!("{:?}", dataset);
    println!("{:?}", dataset.get_tensor(RowType::Train, ColumnType::Target));

    let mut model = Sequential::new();

    model.set_seed(777);

    model.add(ReLU);
    // model.add(hidden_layer_2);
    model.add(ReLU);

    model.fit(&dataset, 2_000, true);
}
