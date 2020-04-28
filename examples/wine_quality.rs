use std::path::Path;

use newron::dataset::Dataset;
use newron::layers::{dense::Dense, relu::ReLU};
use newron::sequential::Sequential;

fn main() {
    let dataset = Dataset::from_csv(Path::new("datasets/winequality-white.csv"), true).unwrap();

    println!("{:?}", dataset);

    let mut model = Sequential::new();

    model.set_seed(777);

    model.add(Dense::new(11, 40));
    model.add(ReLU);

    model.add(Dense::new(40, 1));
    model.add(ReLU);

    model.fit(&dataset, 2_000, true);
}
