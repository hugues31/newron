use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct Softmax;

impl Layer for Softmax {
    fn get_info(&self) -> String {
        format!("Softmax Layer")
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // we use stable softmax instead of classic softmax
        // for computational stability

        let normalized_input = input.normalize_rows();
        let numerator = normalized_input.map(|x| x.exp());
        let denominator = normalized_input.map(|x| x.exp()).get_sum(1);
        numerator / denominator
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor {
        let m = input.shape[0];
        let n = input.shape[1];
        let p = self.forward(input);

        // tensor1 is a 3D tensor (batch size * n x n matrix)
        let mut tensor1: Vec<Tensor> = Vec::new();
        for observation in 0..m {
            let mut data = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    let c_ij = &p.get_value(observation, i) * &p.get_value(observation, j);
                    data.push(c_ij);
                }
            }
            tensor1.push(Tensor { data, shape: vec![n, n] });
        }

        // tensor2 is a 3D tensor (batch size * n x n matrix)
        let mut tensor2: Vec<Tensor> = Vec::new();
        for observations in 0..m {
            let mut data = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    let value = if i == j { p.get_value(observations, i) } else { 0.0 };
                    data.push(value);
                }
            }
            tensor2.push(Tensor { data, shape: vec![n, n] });
        }

        let mut d_softmax: Vec<Tensor> = Vec::new();
        for observations in 0..m {
            d_softmax.push(&tensor2[observations] - &tensor1[observations]);
        }

        let mut data = Vec::new();
        for observation in 0..m {
            for col in 0..n {
                let mut acc = 0.0;
                for col_iter in 0..n {
                    acc += d_softmax[observation].get_value(col, col_iter) * grad_output.get_value(observation, col_iter);
                }
                data.push(acc);
            }  
        }

        Tensor {
            data,
            shape: grad_output.shape.to_vec()
        }
    }
}