use super::{FromGenome, Network, Recurrent, Stateful};
use crate::{Connection, Genome};
use nalgebra as na;

#[cfg(feature = "serialize")]
use crate::serialize::dmatrix;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct NonBias {
    #[cfg_attr(feature = "serialize", serde(with = "dmatrix"))]
    pub y: na::DMatrix<f64>,
    #[cfg_attr(feature = "serialize", serde(with = "dmatrix"))]
    pub w: na::DMatrix<f64>,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl Network for NonBias {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        let mut m_input = na::DMatrix::zeros(1, self.y.ncols());
        m_input.as_mut_slice()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);

        // Preallocate temporary buffers
        let mut temp1 = na::DMatrix::zeros(1, self.y.ncols());
        let mut temp2 = na::DMatrix::zeros(1, self.y.ncols());

        for _ in 0..prec {
            // temp1 = (y + m_input).map(σ)
            temp1.copy_from(&self.y);
            temp1 += &m_input;
            for val in temp1.iter_mut() {
                *val = σ(*val);
            }

            // temp2 = (temp1 * w) * inv
            temp2.gemm(1.0, &temp1, &self.w, 0.0);
            temp2 *= inv;

            self.y.copy_from(&temp2);
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
        let cols = genome.node_count();
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::{self, connection::BWConnection, WConnection};
    use eevee_macros::fn_matrix;
    use rulinalg::matrix::BaseMatrix;

    fn_matrix! {
        C: WConnection | BWConnection,
        G: genome::Recurrent<C>,

        /// weight matrix has correct dimensions for node count
        #[test]
        fn test_matrix_structure() {
            let (genome, _) = G::new(3, 2);
            let nn = NonBias::from_genome(&genome);
            let cols = genome.node_count();

            assert_eq!(nn.y.cols(), cols);
            assert_eq!(nn.y.rows(), 1);
            assert_eq!(nn.w.cols(), cols);
            assert_eq!(nn.w.rows(), cols);
        }

        /// sensory/action ranges map correctly
        #[test]
        fn test_bounds() {
            let (genome, _) = G::new(4, 3);
            let nn = NonBias::from_genome(&genome);

            assert_eq!(nn.sensory.0, genome.sensory().start);
            assert_eq!(nn.sensory.1, genome.sensory().end);
            assert_eq!(nn.action.0, genome.action().start);
            assert_eq!(nn.action.1, genome.action().end);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// disabled connections excluded from weight matrix
        #[test]
        fn test_disabled_connections_excluded() {
            let (mut genome, _) = G::new(2, 2);
            if let Some(c) = genome.connections_mut().first_mut() {
                c.disable();
            }

            let nn = NonBias::from_genome(&genome);
            let cols = genome.node_count();

            assert_eq!(nn.w.data()[0 * cols + 2], 0.0);
        }
    }
}
