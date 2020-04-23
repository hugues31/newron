use std::path::Path;

use newron::dataset::Dataset;

fn main() {
    // Path to a folder containing the 4 files :
    // 1/ train-images-idx3-ubyte
    // 2/ train-labels-idx1-ubyte
    // 3/ t10k-images-idx3-ubyte
    // 4/ t10k-labels-idx1-ubyte
    let path = Path::new("fashion_mnist/");

    let dataset = Dataset::from_ubyte(path).unwrap();

    println!("{:?}", dataset);
}
