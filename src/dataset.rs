use std::fmt;
use std::path::Path;
use std::cmp;
use std::fs::File;
use std::io::Read;

use crate::tensor::Tensor;
use crate::utils;

#[derive(PartialEq)]
pub enum ColumnType {
    Feature, // column is a feature used to train models
    Target,  // column is a target to predict
    Skip     // column not used by the model
}

pub struct Column {
    name: String,
    column_type: ColumnType
}

#[derive(Debug)]
pub enum DatasetError {
    FileNotFound,
    BadFormat(String),
}

/// Use `Dataset` to load your dataset and train
/// a model on it.
pub struct Dataset {
    // Contains all data for dataset
    data: Vec<Vec<f64>>, //TODO: refactor into Vec<Row>
    columns: Vec<Column>
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

        let mut columns = Vec::new();

        // test that all rows in 'data' have equal lengths
        if data.iter().any(|ref v| v.len() != data[0].len()) {
            return Err(DatasetError::BadFormat(format!("All rows must have equal lengths.")));
        }

        // iterate through training features
        for i in 0..cols - 1 {
            columns.push(Column {name: format!("X_{}", i), column_type: ColumnType::Feature})
        }

        columns.push(Column {name: format!("Y"), column_type: ColumnType::Target});

        Ok(Dataset {
            data,
            columns,
        })
    }

    pub fn from_csv() -> Result<Dataset, DatasetError> {
        unimplemented!();
    }

    pub fn from_ubyte(path: &Path) -> Result<Dataset, DatasetError> {
        let mut labels_file = File::open(path.join("train-labels-idx1-ubyte")).unwrap();
        let mut images_file = File::open(path.join("train-images-idx3-ubyte")).unwrap();

        let mut buf = [0u8;4];
        images_file.read(&mut buf).unwrap();
        let magic_number = utils::swap_endian(utils::as_u32_le(&buf));
        assert_eq!(magic_number, 2051, "Incorrect magic number for a image file.");

        let mut buf = [0u8;4];
        labels_file.read(&mut buf).unwrap();
        let magic_number = utils::swap_endian(utils::as_u32_le(&buf));
        assert_eq!(magic_number, 2049, "Incorrect magic number for a label file.");
        
        images_file.read(&mut buf).unwrap();
        let number_images = utils::swap_endian(utils::as_u32_le(&buf));
        
        labels_file.read(&mut buf).unwrap();
        let number_labels = utils::swap_endian(utils::as_u32_le(&buf));

        assert_eq!(number_images, number_labels, "Number of images and label must be identical.");

        images_file.read(&mut buf).unwrap();
        let rows = utils::swap_endian(utils::as_u32_le(&buf)); // =28

        images_file.read(&mut buf).unwrap();
        let cols = utils::swap_endian(utils::as_u32_le(&buf)); // =28
    
        let vector_size = (rows * cols) as usize;

        let mut data: Vec<Vec<f64>> = Vec::new();

        for _ in 0..number_images {
            // read image pixel
            let mut buf = vec![0u8;vector_size];
            images_file.read(&mut buf).unwrap();
            let mut pixels = utils::to_vec_f64(&buf);

            // read label
            let mut label = vec![0u8;1];
            labels_file.read(&mut label).unwrap();
            
            // add row to dataset (pixels + label)
            pixels.append(&mut utils::to_vec_f64(&label));
            data.push(pixels);
        }

        // At this point, the last col is the label
        // We must one-hot-encode it
        let mut dataset = Dataset::from_raw_data(data).unwrap();
        dataset.one_hot_encode(vector_size);

        Ok(dataset)
    }

    /// Use one hot encoding for the column at `index`
    /// Note : the column at `index` is removed and 
    /// replaced with columns containing the one-hot-encoding
    pub fn one_hot_encode(&mut self, index: usize) {
        let distinct_values = self.get_distinct_values(index);
        let number_distinct_values = distinct_values.len();

        // for each row in the dataset
        for row in self.data.iter_mut() {
            let value_to_encode = row[index];
            let position = distinct_values.iter().position(|&x| x == value_to_encode).unwrap();
            // create the base one-hot encoding filled with zeroes
            let mut one_hot = vec![0.0f64; number_distinct_values];
            // Set the one at the correct position
            one_hot[position] = 1.0;
            // add one-hot vector inside the dataset
            row.append(&mut one_hot);
        }

        // add as columns as elements in one-hot vector
        for col in 0..number_distinct_values {
            let name = format!("Y"); // TODO: get unique col name
            let column_type = ColumnType::Target; // TODO: set from method argument
            self.columns.push(Column {name, column_type});
        }

        // remove the old column
        self.remove_column(index)
    }

    /// Remove column at `index`
    pub fn remove_column(&mut self, index: usize) {
        // remove column metadata
        self.columns.remove(index);

        // remove the specified column in data
        for row in self.data.iter_mut() {
            row.remove(index);
        }
    }

    /// Get all distinct values for column at `index` (sorted)
    pub fn get_distinct_values(&self, index: usize) -> Vec<f64> {
        let mut result = Vec::new();
        for row in &self.data {
            let value = row[index];
            if !result.contains(&value) {
                result.push(value);
            }
        }
        // sort values (float cannot be perfectly compared so we use partial_cmp)
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result
    }

    /// Get the train features
    pub fn get_train(&self) -> Tensor {
        let rows = self.data.len();
        let cols = self.count_column_type(ColumnType::Feature);
        let shape = vec![rows, cols];

        let mut train = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                train.push(self.data[i][j]);
            }
        }

        Tensor::new(train, shape)
    }

    /// Get the target features
    pub fn get_target(&self) -> Tensor {
        let rows = self.data.len();
        let cols = self.count_column_type(ColumnType::Target);
        let shape = vec![rows, cols];

        let mut target = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                target.push(self.data[i][j]);
            }
        }

        Tensor::new(target, shape)
    }

    // Count the number of columns in the dataset matching the type `col_type`
    fn count_column_type(&self, col_type: ColumnType) -> usize {
        self.columns.iter().filter(|&n| n.column_type == col_type).count()
    }
}

// Implement Debug
impl fmt::Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dataset")
            .field("data", &format_args!("\n{}", self))
            .field("Observation(s)", &self.data.len())
            .field("Feature(s)", &self.count_column_type(ColumnType::Feature))
            .field("Target(s)", &self.count_column_type(ColumnType::Target))
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

        // Construct table
        // 4 rows maximum
        let rows = cmp::min(self.data.len(), 4);
        // 12 cols maximum
        let cols = cmp::min(self.data[0].len(), 12);

        // Construct header
        let mut headers = Vec::new();
        for c in &self.columns {
            headers.push(c.name.to_string());
        }
        let header_string = headers[0..cols].join(sep);
        result.push_str(&header_string);

        for row in 0..rows {
            let mut temp_row: Vec<String> = Vec::new();
            for col in 0..cols {
                let col_len = headers[col].len();
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

