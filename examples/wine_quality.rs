use std::path::Path;

use newron::dataset::Dataset;
use newron::layers::LayerEnum::*;
use newron::optimizers::sgd::SGD;
use newron::sequential::Sequential;
use newron::loss::{mse::MSE};
use newron::metrics::MetricEnum;

fn main() {
    let dataset = Dataset::from_csv(Path::new("datasets/winequality-white.csv"), true).unwrap();

    println!("{:?}", dataset);

    let mut model = Sequential::new();

    model.set_seed(42);

    model.add(Dense {
        input_units: dataset.get_number_features(),
        output_units: 100
    });
    
    model.add(ReLU);

    model.add(Dense {
        input_units: 100,
        output_units: dataset.get_number_targets()
    });

    model.compile(MSE{},
        SGD::new(0.0002),
        vec![MetricEnum::Accuracy]);

    model.fit(&dataset, 200, true);
}
