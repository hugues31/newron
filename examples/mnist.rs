use std::path::Path;

use newron::dataset::Dataset;
use newron::layers::{dense::Dense, relu::ReLU, softmax::Softmax};
use newron::loss::{categorical_entropy::CategoricalEntropy};
use newron::metrics::Metrics;
use newron::sequential::Sequential;
use newron::optimizers::optimizer::Optimizer;

fn main() {
    // Path to a folder containing the 4 files :
    // 1/ train-images-idx3-ubyte
    // 2/ train-labels-idx1-ubyte
    // 3/ t10k-images-idx3-ubyte
    // 4/ t10k-labels-idx1-ubyte
    let path = Path::new("datasets/fashion_mnist/");

    let dataset = Dataset::from_ubyte(path).unwrap();
    println!("{:?}", dataset);

    let mut model = Sequential::new();
    model.add(Dense::new(dataset.get_number_features(), 40));
    model.add(ReLU);
    model.add(Dense::new(40, dataset.get_number_targets()));
    model.add(Softmax);

    model.summary();

    model.compile(CategoricalEntropy{},
              Optimizer::SGD,
              vec![Metrics::Accuracy]);

    model.fit(&dataset, 2_000, true);
}
