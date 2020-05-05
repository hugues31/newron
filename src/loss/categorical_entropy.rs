use crate::{tensor::Tensor, loss::loss::Loss, utils};
pub struct CategoricalEntropy {}

impl Loss for CategoricalEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f64 {
        let m = y_true.shape[0];

        let indices = utils::one_hot_encoded_tensor_to_indices(y_true);

        let mut p = Vec::new();
        for (row, indice) in indices.iter().enumerate() {
            p.push(y_pred.get_value(row, *indice));
        }

        let log_likelihood: Vec<f64> = p.iter().map(|x| -(x.ln())).collect();

        log_likelihood.iter().sum::<f64>() / m as f64
    }

    fn compute_loss_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        let rows = y_true.shape[0];
        let cols = y_true.shape[1];

        let indices = utils::one_hot_encoded_tensor_to_indices(y_true);

        let mut data: Vec<f64> = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let indice = indices[row];
                let mut value = y_pred.get_value(row, col);
                if col == indice { value -= 1.0 }
                value /= rows as f64;
                data.push(value);

            }
        }

        Tensor {
            data,
            shape: vec![rows, cols]
        }

    }
}