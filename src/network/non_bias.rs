use super::{FromGenome, Network, Recurrent, Stateful};
use crate::{
    serialize::{deserialize_matrix_flat, deserialize_matrix_square, serialize_matrix},
    Connection, Genome,
};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct NonBias {
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub y: na::DMatrix<f64>,
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_square"
    )]
    pub w: na::DMatrix<f64>,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl Network for NonBias {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        let mut m_input = na::DMatrix::zeros(1, self.y.ncols());
        m_input.as_mut_slice()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            self.y = ((&self.y + &m_input).map(&σ) * &self.w).map(|v| v * inv);
        }
    }

    fn flush(&mut self) {
        self.y = na::DMatrix::zeros(1, self.y.ncols());
    }

    fn output(&self) -> &[f64] {
        &self.y.as_slice()[self.action.0..self.action.1]
    }
}

impl Recurrent for NonBias {}

impl Stateful for NonBias {}

impl<C: Connection, G: Genome<C>> FromGenome<C, G> for NonBias {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.nodes().len();
        Self {
            y: na::DMatrix::zeros(1, cols),
            w: {
                let mut w = vec![0.; cols * cols];
                for c in genome.connections().iter().filter(|c| c.enabled()) {
                    w[c.from() * cols + c.to()] = c.weight();
                }
                na::DMatrix::from_row_slice(cols, cols, &w)
            },
            sensory: (genome.sensory().start, genome.sensory().end),
            action: (genome.action().start, genome.action().end),
        }
    }
}
