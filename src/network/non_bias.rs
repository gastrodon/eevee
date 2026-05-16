use super::{FromGenome, Network, Recurrent, Stateful};
use crate::{Connection, Genome};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

#[derive(Debug)]
pub struct NonBias {
    pub y: Matrix<f64>,
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
        let cols = genome.node_count();
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
