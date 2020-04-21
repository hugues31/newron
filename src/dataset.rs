use std::path::Path;
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
    // Indice of training features (cols) to use for training
    train_cols: Vec<usize>,
    // Indice of target(s) colunm(s) to use
    target_cols: Vec<usize>
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
        let mut train_cols = Vec::new();
        let mut target_cols = Vec::new();
        
        // iterate through training features
        for i in 0..cols - 1 {
            header.push(format!("{}_", i));
            train_cols.push(i);
        }

        header.push(format!("Y"));
        target_cols.push(cols - 1);

        Ok(Dataset{
            data,
            header,
            train_cols,
            target_cols
        })
    }

    pub fn from_csv() -> Result<Dataset, DatasetError> {
        unimplemented!();
    }

    pub fn get_train(&self) -> Tensor {
        let rows = self.data.len();
        let cols = self.train_cols.len();
        let shape = vec![rows, cols];

        let mut train = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
               train.push(self.data[i][j]);
            }
        }

        Tensor::new(
            train,
            shape
        )
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

        Tensor::new(
            target,
            shape
        )
    }
}