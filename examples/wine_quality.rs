use std::path::Path;

use newron::dataset::Dataset;

fn main() {
    let dataset = Dataset::from_csv(Path::new("datasets/winequality-white.csv"), true).unwrap();

    println!("{:?}", dataset);
}
