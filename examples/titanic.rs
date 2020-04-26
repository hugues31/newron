use std::path::Path;

use newron::dataset::Dataset;

fn main() {
    let path = Path::new("/home/brieuc/Downloads/titanic");

    let dataset = Dataset::from_csv(path).unwrap();

    println!("{:?}", dataset);
}
