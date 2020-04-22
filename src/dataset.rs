use std::fmt;
use std::path::Path;
use std::cmp;

use crate::tensor::Tensor;

#[derive(Debug)]
pub enum DatasetError {
    FileNotFound,
    BadFormat,
}

/// Use `Dataset` to load your dataset and train
/// a model on it.
pub struct Dataset {
    // Contains all data for dataset
    data: Vec<Vec<f64>>,
    // Contains name of the columns
    header: Vec<String>,
    // Indice of features (cols) to use for training
    feature_cols: Vec<usize>,
    // Indice of target(s) colunm(s) to use
    target_cols: Vec<usize>,
}

impl Dataset {
    /// Load a dataset from a Vector of Vector of floats.
    /// By default, the last colunm is use as a target and the others as
    /// training features. Use `set_train_cols` and `set_target_cols` to
    /// change this behaviour.
    /// Header is automatically generated : 'X_0', 'X_1', ..., 'Y'. Use
    /// `set_header` to change it.
    pub fn from_raw_data(data: Vec<Vec<f64>>) -> Result<Dataset, DatasetError> {
        let cols = data[0].len();

        let mut header = Vec::new();
        let mut feature_cols = Vec::new();
        let mut target_cols = Vec::new();
        // iterate through training features
        for i in 0..cols - 1 {
            header.push(format!("X_{}", i));
            feature_cols.push(i);
        }

        header.push(format!("Y"));
        target_cols.push(cols - 1);

        Ok(Dataset {
            data,
            header,
            feature_cols,
            target_cols,
        })
    }

    pub fn from_csv() -> Result<Dataset, DatasetError> {
        unimplemented!();
    }

    pub fn get_train(&self) -> Tensor {
        let rows = self.data.len();
        let cols = self.feature_cols.len();
        let shape = vec![rows, cols];

        let mut train = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                train.push(self.data[i][j]);
            }
        }

        Tensor::new(train, shape)
    }

    pub fn get_target(&self) -> Tensor {
        let rows = self.data.len();
        let cols = self.target_cols.len();
        let shape = vec![rows, cols];

        let mut target = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                target.push(self.data[i][j]);
            }
        }

        Tensor::new(target, shape)
    }
}

// Implement Debug
impl fmt::Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dataset")
            .field("data", &format_args!("{}", self))
            .field("Observation(s)", &self.data.len())
            .field("Feature(s)", &self.feature_cols.len())
            .field("Target(s)", &self.target_cols.len())
            .finish()
    }
}

// Implement Display
impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Example :
        // X_0 | X_1 | X_2 | Y
        // 1   | 0   | 0   | 1
        let sep = " | "; // separator
        let mut result = String::new();

        // Construct header
        let header_string = self.header.join(sep);
        result.push_str(&header_string);

        // Construct table
        let rows = self.data.len();
        let cols = self.data[0].len();

        // 3 rows maximum to be displayed
        for row in 0..cmp::min(rows,3) {
            let mut temp_row: Vec<String> = Vec::new();
            for col in 0..cols {
                let col_len = self.header[col].len();
                let mut value = self.data[row][col].to_string();
                let value_len = value.len();
                // if we must truncate value
                if value_len > col_len {
                    value = value[0..col_len].to_string();
                }
                // otherwise, we must pad with whitespaces
                else {
                    value = value + &" ".repeat(col_len - value_len);
                }

                temp_row.push(value);
            }
            result.push_str("\n");
            result.push_str(&temp_row.join(sep));
        }
        
        write!(f, "{}", result)
    }
}
