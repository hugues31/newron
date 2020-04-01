pub fn relu(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

pub fn relu2deriv(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}