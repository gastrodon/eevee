use super::{FromGenome, Network, Recurrent, Stateful};
use crate::{Connection, Genome};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NonBias {
    #[cfg_attr(
        feature = "serialize",
        serde(
            serialize_with = "crate::serialize::serialize_matrix",
            deserialize_with = "crate::serialize::deserialize_matrix_flat"
        )
    )]
    pub y: Matrix<f64>,
    #[cfg_attr(
        feature = "serialize",
        serde(
            serialize_with = "crate::serialize::serialize_matrix",
            deserialize_with = "crate::serialize::deserialize_matrix_square"
        )
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

impl<C: Connection, G: Genome<C>> FromGenome<C, G> for NonBias {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.nodes().len();
        Self {
            y: Matrix::zeros(1, cols),
            w: {
                let mut w = vec![0.; cols * cols];
                for c in genome.connections().iter().filter(|c| c.enabled()) {
                    w[c.from() * cols + c.to()] = c.weight();
                }
                Matrix::new(cols, cols, w)
            },
            sensory: (genome.sensory().start, genome.sensory().end),
            action: (genome.action().start, genome.action().end),
        }
    }
}
