#[cfg(test)]
mod tests {
    use newron::activation::*;
    
    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid_activation(0.), 0.5)
    }
    #[test]
    fn test_sigmoid2deriv() {
        assert_eq!(sigmoid_deriv(0.), 0.25)
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu_activation(0.), 0.)
    }
    #[test]
    fn test_relu2deriv() {
        assert_eq!(relu_deriv(0.), 1.)
    }
}
