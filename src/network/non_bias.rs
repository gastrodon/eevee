use super::{Network, Recurrent, Stateful};
use crate::serialize::{deserialize_matrix_flat, deserialize_matrix_square, serialize_matrix};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct NonBias {
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub y: Matrix<f64>,
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_square"
    )]
    pub w: Matrix<f64>,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl Network for NonBias {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            self.y = ((&self.y + &m_input).apply(&σ) * &self.w).apply(&|v| v * inv);
        }
    }

    fn flush(&mut self) {
        self.y = Matrix::zeros(1, self.y.cols());
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}

impl Recurrent for NonBias {}

impl Stateful for NonBias {}
