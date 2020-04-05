use newron::tensor::Tensor;

fn main() {
    let a = Tensor::new(vec![1.5], vec![]);
    let b = Tensor::new(vec![2.0], vec![]);
    
    let c = Tensor::new(vec![1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0], vec![2,3]);

    let d = Tensor::new(vec![3.0, 4.0,
                             5.0, 6.0,
                             7.0, 8.0], vec![3,2]);

    // println!("a+b: {:?}", a+b);

    // let truc: Vec<f64> = ;

    // println!("truc : {:?}", truc);

    let e = vec![1.0];
    let f = vec![2.0];

    // println!("{:?}", b.data);
    println!("--> {:?}", c * d);
}
