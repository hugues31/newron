use std::path::Path;

use newron::dataset::Dataset;
use newron::layers::{dense::Dense, relu::ReLU};
use newron::sequential::Sequential;

fn main() {
    let dataset = Dataset::from_csv(Path::new("datasets/winequality-white.csv"), true).unwrap();

    println!("{:?}", dataset);

    let mut model = Sequential::new();

    model.set_seed(42);

    model.add(Dense::new(dataset.get_number_features(), 10));
    model.add(ReLU);

    model.add(Dense::new(10, dataset.get_number_targets()));
    model.add(ReLU);

    model.fit(&dataset, 200, true);
}
