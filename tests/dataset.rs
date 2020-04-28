#[cfg(test)]
mod dataset_tests {
    use newron::dataset::Dataset;
    use std::path::Path;
    #[test]
    // This test asserts a good implementation of
    // debug + display trait + loading from raw data
    fn test_dataset_debug_trait() {
        let dataset = Dataset::from_raw_data(vec![
            //   X_0, X_1, X_2, Y
            vec![1.0, 0.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0],
        ])
        .unwrap();

        let result = "X_0 | X_1 | X_2 | Y\n\
                      1   | 0   | 1   | 1\n\
                      0   | 1   | 1   | 1\n\
                      0   | 0   | 1   | 0\n\
                      1   | 1   | 1   | 0\n\
                      Observation(s): 4 (4 train + 0 test) \n\
                      Feature(s): 3\n\
                      Target(s): 1\n";

        assert_eq!(format!("{:?}", dataset), result);
    }

    #[test]
    fn test_load_csv() {
        let dataset = Dataset::from_csv(Path::new("datasets/winequality-white.csv"), true).unwrap();

        assert_eq!(dataset.get_number_features(), 11);
        assert_eq!(dataset.get_number_targets(), 1);
        assert_eq!(dataset.get_row_count(), 4898);

    }
}
