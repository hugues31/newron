#[cfg(test)]
mod tensor_tests {
    use newron::tensor::Tensor;
    
    #[test]
    fn test_0d_add() {
        // Scalar
        let a = Tensor::new(vec![1.0], vec![]);
        let b = Tensor::new(vec![2.0], vec![]);

        let c = Tensor::new(vec![3.0], vec![]);

        assert_eq!(a + b, c);
    }

    #[test]
    fn test_1d_add() {
        // Vector
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);

        let c = Tensor::new(vec![4.0, 6.0], vec![2]);

        assert_eq!(a + b, c);
    }

    #[test]
    fn test_2d_add() {
        // 2x2 matrix
        let a = Tensor::new(vec![1.0, 2.0,
                                 3.0, 4.0],
                                 vec![2, 2]);

        let b = Tensor::new(vec![5.0, 6.0,
                                 7.0, 8.0],
                                 vec![2, 2]);

        let c = Tensor::new(vec![6.0,   8.0,
                                 10.0, 12.0],
                                 vec![2, 2]);

        assert_eq!(a + b, c);
    }

    #[test]
    fn test_2d_1d_add() {
        // 2x4 matrix
        let a = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![2, 4]);

        // 1x4 matrix
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 4]);

        let result = Tensor::new(vec![1.0, 3.0, 5.0, 7.0, 5.0, 7.0, 9.0, 11.0], vec![2, 4]);

        assert_eq!(a + b, result)

    }

    #[test]
    fn test_get_row() {
        // 2x2 matrix
        let a = Tensor::new(vec![1.0, 2.0,
                                 3.0, 4.0],
                                 vec![2, 2]);
        
        let test_row = Tensor::new(vec![3.0, 4.0], vec![1, 2]);

        assert_eq!(a.get_row(1), test_row);
    }

    #[test]
    fn test_get_rows() {
        // 4x2 matrix
        let a = Tensor::new(vec![1.0, 2.0,
                                 3.0, 4.0,
                                 5.0, 6.0,
                                 7.0, 8.0],
                                 vec![4, 2]);
        
        let test_row = Tensor::new(vec![3.0, 4.0, 7.0, 8.0], vec![2, 2]);

        assert_eq!(a.get_rows(&[1, 3]), test_row);
    }

    #[test]
    fn test_dot_sum_product() {
        // test if dot() acts as the dot() function in Numpy
        let a = Tensor::new(vec![1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0],
                                 vec![2, 3]);

        let b = Tensor::new(vec![1.0, 0.5], vec![1, 2]);

        let result = Tensor::new(vec![3.0, 4.5, 6.0], vec![1, 3]);

        assert_eq!(a.dot(&b), result);
    }

    #[test]
    fn test_dot_2d_tensor() {
        let a = Tensor::new(vec![1.0,2.0,3.0,4.0,5.0,6.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0], vec![3, 4]);

        let result = Tensor::new(vec![6.0, 12.0, 18.0, 24.0, 15.0, 30.0, 45.0, 60.0], vec![2, 4]);

        assert_eq!(a.dot(&b), result);
    }

    #[test]
    fn test_map() {
        fn relu() -> fn(f64) -> f64 {
            |x| if x < 0.0 { 0.0 } else { x }
        }
        
        let a = Tensor::new(vec![1.0, -2.0,
                                -3.0,  4.0],
                                 vec![2, 2]);

        let result = Tensor::new(vec![1.0, 0.0,
                                      0.0, 4.0],
                                      vec![2, 2]);
        
        assert_eq!(a.map(relu()), result);
    }

    #[test]
    fn test_sub_assign() {
        let mut a = Tensor::new(vec![1.0, -2.0,
                                    -3.0,  4.0],
                                     vec![2, 2]);

        let b = Tensor::new(vec![1.0, 2.0,
                                 1.0, 2.0],
                                 vec![2, 2]);

        let result = Tensor::new(vec![0.0, -4.0,
                                     -4.0, 2.0],
                                      vec![2, 2]);
        
        a -= b;

        assert_eq!(a, result);
    }


    #[test]
    fn test_mask() {
         let mask = Tensor::mask(&vec![10, 10], 0.4, 777);
         assert_eq!(mask.data.iter().sum::<f64>() as usize, 100);
    }

    #[test]
    fn test_get_mean_axis_0() {
        let a = Tensor::new(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0], vec![2,3]);
        
        let result = Tensor::new(vec![2.5, 3.5, 4.5], vec![1, 3]);

        assert_eq!(a.get_mean(0), result);
    }

    #[test]
    fn test_get_mean_axis_1() {
        let a = Tensor::new(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0], vec![2,3]);
        
        let result = Tensor::new(vec![2.0, 5.0], vec![1, 2]);

        assert_eq!(a.get_mean(1), result);
    }

    #[test]
    fn test_get_max_axis_0() {
        let a = Tensor::new(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0], vec![2,3]);
        
        let result = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]);

        assert_eq!(a.get_max(0), result);
    }

    #[test]
    fn test_get_max_axis_1() {
        let a = Tensor::new(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0], vec![2,3]);
        
        let result = Tensor::new(vec![3.0,
                                                    6.0], vec![2, 1]);

        assert_eq!(a.get_max(1), result);
    }

    #[test]
    fn test_get_sum_axis_0() {
        let a = Tensor::new(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0], vec![2,3]);
        
        let result = Tensor::new(vec![5.0, 7.0, 9.0], vec![1, 3]);

        assert_eq!(a.get_sum(0), result);
    }

    #[test]
    fn test_get_sum_axis_1() {
        let a = Tensor::new(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0], vec![2,3]);
        
        let result = Tensor::new(vec![6.0,
                                                    15.0], vec![2, 1]);

        assert_eq!(a.get_sum(1), result);
    }

    #[test]
    fn test_divide() {
        let a = Tensor::new(vec![4.0, 6.0, 8.0, 10.0], vec![1, 4]);
        let b = Tensor::new(vec![4.0, 3.0, 5.0, 5.0], vec![1, 4]);

        let result = Tensor::new(vec![1.0, 2.0, 1.6, 2.0], vec![1, 4]);

        assert_eq!(a / b, result);
    }

    #[test]
    fn test_get_transpose() {
        let a = Tensor::new(vec![
             4.0,  6.0,  8.0,
            10.0, 12.0, 14.0], vec![2, 3]);

        let a_t = a.get_transpose();

        let result = Tensor::new(vec![
            4.0, 10.0,
            6.0, 12.0,
            8.0, 14.0,
        ], vec![3, 2]);

        assert_eq!(a_t, result);
    }

}
