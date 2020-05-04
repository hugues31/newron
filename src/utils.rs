// Some utility functions
use crate::tensor::Tensor;

// Invert integer
pub(crate) fn swap_endian(val: u32) -> u32 {
    let result = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (result << 16) | (result >> 16);
}

// Convert an array of 4 u8 into a u32
pub(crate) fn as_u32_le(array: &[u8; 4]) -> u32 {
    ((array[0] as u32) <<  0) +
    ((array[1] as u32) <<  8) +
    ((array[2] as u32) << 16) +
    ((array[3] as u32) << 24)
}

// Convert a vec of u8 into a vec of f64
// Slow implementation (creates a copy)
// but fast enough for file conversion
pub(crate) fn to_vec_f64(data: &Vec<u8>) -> Vec<f64> {
    let mut result = Vec::new();
    for el in data {
        result.push(*el as f64);
    }
    result
}

// Return a rounded version of the vector `data` to
// the number of `decimal_places` specified
pub fn round_vector(data: Vec<f64>, decimal_places: usize) -> Vec<f64> {
    data.iter().map(|x| round_f64(*x, decimal_places)).collect()
}

pub fn round_f64(value: f64, decimal_places: usize) -> f64 {
    let decimal_places = 10.0f64.powf(decimal_places as f64);
    (value * decimal_places).round() / decimal_places
}

// Return a list of indices of the maximum value found for each row of the tensor `data`
// (similar to argmax in numpy)
pub(crate) fn one_hot_encoded_tensor_to_indices(data: &Tensor) -> Vec<usize> {
    let rows = data.shape[0];
    let cols = data.shape[1];

    let mut result = Vec::new();

    for row in 0..rows {
        let mut maximum = data.get_value(row, 0);
        let mut indice = 0;
        for col in 1..cols {
            let value = data.get_value(row, col);
            if value > maximum {
                maximum = value;
                indice = col;
            }
        }
        result.push(indice);
    }

    result
}