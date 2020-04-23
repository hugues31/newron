// Some utility functions

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