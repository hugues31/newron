use std::path::Path;

use newron::dataset::Dataset;
use newron::layers::LayerEnum::*;
use newron::loss::{categorical_entropy::CategoricalEntropy};
use newron::metrics::MetricEnum;
use newron::sequential::Sequential;
use newron::optimizers::sgd::SGD;

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
    model.set_seed(99);

    model.add(Dense {
        input_units: dataset.get_number_features(),
        output_units: 100
    });
    
    model.add(ReLU);

    model.add(Dense {
        input_units: 100,
        output_units: dataset.get_number_targets()
    });

    model.compile(CategoricalEntropy{},
              SGD::new(0.2),
              vec![MetricEnum::Accuracy]);

    model.summary();

    model.fit(&dataset, 20, true);
}
