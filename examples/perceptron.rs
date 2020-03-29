use nalgebra::{DMatrix};

use newron;

type Matrix = DMatrix::<f64>;

fn relu(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}


fn relu2deriv(x: f64) -> f64 {
    if x > 1.0 {
        1.0
    } else {
        0.0
    }
}


fn main() {
    let sample_len = 3; // 3 values by row
    let data = Matrix::from_row_slice(4,sample_len,&[
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0]);

    let result = Matrix::from_row_slice(4,1,&[1.0, 1.0, 0.0, 0.0]);

    let alpha = 0.2;
    let hidden_size = 4;

    let mut weights_0_1: Matrix = Matrix::new_random(sample_len, hidden_size).map(|x| 2.0 * x - 1.0);
    let mut weights_1_2: Matrix = Matrix::new_random(hidden_size, 1).map(|x| 2.0 * x - 1.0);

    for iteration in 0..100_000 {
        let mut layer_2_error = 0.0;

        // iterate throught samples
        for i in 0..data.shape().0 {
            let layer_0 = data.row(i);
            let layer_1 = (layer_0*&weights_0_1).map(relu);
            let layer_2 = &layer_1 * &weights_1_2;
            layer_2_error += (&layer_2 - result.row(i))[0].powi(2);
            let layer_2_delta = &layer_2 - result.row(i);

            let layer_1_delta_dot = &layer_2_delta * &weights_1_2.transpose();
            let layer_1_delta = layer_1_delta_dot.component_mul(&layer_1.map(relu2deriv));
  
            weights_1_2 -= alpha * layer_1.transpose() * &layer_2_delta;
            weights_0_1 -= alpha * layer_0.transpose() * &layer_1_delta;
        }

        if iteration % 10 == 9 {
            println!("Error : {}", layer_2_error);
        }
    }
}
