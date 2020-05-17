use std::fmt;
use std::path::Path;
use std::cmp;
use std::fs::File;
use std::io::{Read, BufReader, BufRead};
use std::str::FromStr;

use crate::tensor::Tensor;
use crate::utils;

#[derive(PartialEq, Debug)]
pub enum ColumnType {
    Feature, // column is a feature used to train models
    Target,  // column is a target to predict
    Skip     // column not used by the model
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum RowType {
    Train,  // row is used for training
    Test,   // row is preserved for test
    Skip    // row is ignored
}

#[derive(Debug)]
pub struct ColumnMetadata {
    name: String,
    column_type: ColumnType
}

#[derive(Debug)]
pub struct Row {
    data: Vec<f64>,
    row_type: RowType
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
    data: Vec<Row>,
    columns_metadata: Vec<ColumnMetadata>
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

        let mut columns_metadata = Vec::new();

        // test that all rows in 'data' have equal lengths
        if data.iter().any(|ref v| v.len() != data[0].len()) {
            return Err(DatasetError::BadFormat(format!("All rows must have equal lengths.")));
        }

        // iterate through training features
        for i in 0..cols - 1 {
            columns_metadata.push(ColumnMetadata {name: format!("X_{}", i), column_type: ColumnType::Feature})
        }

        columns_metadata.push(ColumnMetadata {name: format!("Y"), column_type: ColumnType::Target});

        let mut rows = Vec::new();
        for el in data {
            rows.push(Row{data: el, row_type: RowType::Train});
        }

        Ok(Dataset {
            data: rows,
            columns_metadata,
        })
    }

    /// Load a CSV from the `path` specified. If `header` is true, the first
    /// line defines the header.
    pub fn from_csv(path: &Path, header: bool) -> Result<Dataset, DatasetError> {
        let delimiter = ";";

        let input_file = File::open(path).unwrap();
        let buffered = BufReader::new(input_file);

        let mut data: Vec<Vec<f64>> = Vec::new();

        for (i, line) in buffered.lines().enumerate() {
            let l = line.unwrap();
            let l = l.split(&delimiter);
            let row_vec_str: Vec<&str> = l.collect();

            if i == 0 {
                continue; // TODO: implement header
            }

            let row_vec_f64: Vec<f64> = row_vec_str.iter().map(|x| f64::from_str(x).unwrap()).collect();
            data.push(row_vec_f64);
        }

        let dataset = Dataset::from_raw_data(data).unwrap();
        Ok(dataset)
    }

    fn load_ubyte(path: &Path, dataset: String) -> Result<Dataset, DatasetError> {
        let mut labels_file = File::open(path.join(format!("{}-labels-idx1-ubyte", dataset))).unwrap();
        let mut images_file = File::open(path.join(format!("{}-images-idx3-ubyte", dataset))).unwrap();

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
            pixels = pixels.into_iter().map(|x| x / 255.0).collect();
            // read label
            let mut label = vec![0u8;1];
            labels_file.read(&mut label).unwrap();
            
            // add row to dataset (pixels + label)
            pixels.append(&mut utils::to_vec_f64(&label));
            data.push(pixels);
        }
        
        let mut dataset = Dataset::from_raw_data(data).unwrap();

        // At this point, the last col is the label
        // We must one-hot-encode it
        dataset.one_hot_encode(vector_size);

        Ok(dataset)
    }

    pub fn from_ubyte(path: &Path) -> Result<Dataset, DatasetError> {
        let train_dataset = Dataset::load_ubyte(path, "train".to_string()).unwrap();
        let mut test_dataset = Dataset::load_ubyte(path, "t10k".to_string()).unwrap();
        // set all rows to "test" type for the test_dataset
        test_dataset.set_all_rows_type(RowType::Test);
        // Add train dataset inside test dataset
        test_dataset.concatenate(train_dataset);
        Ok(test_dataset)
    }

    /// Set the row type to `row_type` at the `index` specified.
    pub fn set_row_type(&mut self, row_type: RowType, index: usize) {
        self.data[index].row_type = row_type;
    }

    /// Set the row type to `row_type` for all the rows.
    pub fn set_all_rows_type(&mut self, row_type: RowType) {
        for row in self.data.iter_mut() {
            row.row_type = row_type;
        }
    }

    /// Concatenate the `other` dataset inside `self`.
    pub fn concatenate(&mut self, other: Dataset) {
        // TODO: check that the columns are the same
        self.data.extend(other.data);
    }

    /// Use one hot encoding for the column at `index`
    /// Note : the column at `index` is removed and 
    /// replaced with columns containing the one-hot-encoding
    pub fn one_hot_encode(&mut self, index: usize) {
        let distinct_values = self.get_distinct_values(index);
        let number_distinct_values = distinct_values.len();

        // for each row in the dataset
        for row in self.data.iter_mut() {
            let value_to_encode = row.data[index];
            let position = distinct_values.iter().position(|&x| x == value_to_encode).unwrap();
            // create the base one-hot encoding filled with zeroes
            let mut one_hot = vec![0.0f64; number_distinct_values];
            // Set the one at the correct position
            one_hot[position] = 1.0;
            // add one-hot vector inside the dataset
            row.data.append(&mut one_hot);
        }

        // add as columns as elements in one-hot vector
        for _col in 0..number_distinct_values {
            let name = format!("Y"); // TODO: get unique col name
            let column_type = ColumnType::Target; // TODO: set from method argument
            self.columns_metadata.push(ColumnMetadata {name, column_type});
        }

        // remove the old column
        self.remove_column(index)
    }

    /// Remove column at `index`
    pub fn remove_column(&mut self, index: usize) {
        // remove column metadata
        self.columns_metadata.remove(index);

        // remove the specified column in data
        for row in self.data.iter_mut() {
            row.data.remove(index);
        }
    }

    /// Get all distinct values for column at `index` (sorted)
    pub fn get_distinct_values(&self, index: usize) -> Vec<f64> {
        let mut result = Vec::new();
        for row in &self.data {
            let value = row.data[index];
            if !result.contains(&value) {
                result.push(value);
            }
        }
        // sort values (float cannot be perfectly compared so we use partial_cmp)
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result
    }

    /// Get the tensor from the dataset with the `row_type` and 
    /// `column_type` specified.
    pub fn get_tensor(&self, row_type: RowType, col_type: ColumnType) -> Tensor {
        let rows = self.count_row_type(&row_type);
        let cols = self.count_column_type(&col_type);
        let shape = vec![rows, cols];

        let mut col_indexes = Vec::new();
        for (i, col) in self.columns_metadata.iter().enumerate() {
            if col.column_type == col_type {
                col_indexes.push(i);
            }
        }

        let mut result = Vec::new();
        for row in &self.data {
            if row.row_type == row_type {
                for col in &col_indexes {
                    result.push(row.data[*col]);
                }
            }
        }

        Tensor::new(result, shape)
    }

    // Count the number of columns in the dataset matching the type `col_type`
    fn count_column_type(&self, col_type: &ColumnType) -> usize {
        self.columns_metadata.iter().filter(|&n| n.column_type == *col_type).count()
    }

    pub fn count_row_type(&self, row_type: &RowType) -> usize {
        self.data.iter().filter(|&r| r.row_type == *row_type).count()
    }

    pub fn get_number_features(&self) -> usize {
        self.count_column_type(&ColumnType::Feature)
    }

    pub fn get_number_targets(&self) -> usize {
        self.count_column_type(&ColumnType::Target)
    }

    pub fn get_row_count(&self) -> usize {
        self.data.len()
    }
}

// Implement Debug
impl fmt::Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\n\
        Observation(s): {} ({} train + {} test) \n\
        Feature(s): {}\n\
        Target(s): {}\n\
        ", self, &self.get_row_count(),
        &self.count_row_type(&RowType::Train),
        &self.count_row_type(&RowType::Test),
        &self.get_number_features(),
        &self.get_number_targets()
    )
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
        let cols = cmp::min(self.data[0].data.len(), 12);

        // Construct header
        let mut headers = Vec::new();
        for c in &self.columns_metadata {
            headers.push(c.name.to_string());
        }
        let header_string = headers[0..cols].join(sep);
        result.push_str(&header_string);

        for row in 0..rows {
            let mut temp_row: Vec<String> = Vec::new();
            for col in 0..cols {
                let col_len = headers[col].len();
                let mut value = self.data[row].data[col].to_string();
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

