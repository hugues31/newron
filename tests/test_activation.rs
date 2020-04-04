#[cfg(test)]
mod tests {
    use newron::activation::*;
    
    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.), 0.5)
    }
    #[test]
    fn test_sigmoid2deriv() {
        assert_eq!(sigmoid2deriv(0.), 0.25)
    }
}
