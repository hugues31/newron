#[cfg(test)]
mod tensor_tests {
    use newron::tensor::Tensor;
    
    #[test]
    fn test_0d_add() {
        let a = Tensor::new(vec![1.0], vec![]);
        let b = Tensor::new(vec![2.0], vec![]);

        let c = Tensor::new(vec![3.0], vec![]);

        assert_eq!(a + b, c);
    }

    #[test]
    fn test_1d_add() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);

        let c = Tensor::new(vec![4.0, 6.0], vec![2]);

        assert_eq!(a + b, c);
    }

    #[test]
    fn test_2d_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let c = Tensor::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2]);

        assert_eq!(a + b, c);
    }
}