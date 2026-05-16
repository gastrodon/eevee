use super::{FromGenome, Recurrent, Stateful};
use crate::{Connection, Genome, Network};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

/// A stateful NN who receives input continuously, useful for realtime problems
/// and genomes whos connections may be recurrent.
///
/// Implementation based on the network described by
/// on the dynamics of small continuous-time recurrent neural networks (beer 1995)
/// and with some code stolen from [TLmaK0's neat implentation](https://github.com/TLmaK0/rustneat)
#[derive(Debug)]
pub struct Continuous {
    /// 1d state of neurons 0-N
    pub y: Matrix<f64>,
    /// 1d bias of neurons 0-N
    pub θ: Matrix<f64>,
    /// 1d membrane resistance time constant
    pub τ: Matrix<f64>,
    /// Nd weights between neurons, indexed as [from, to]
    pub w: Matrix<f64>,
    /// Range of input neurons, indexing into y
    pub sensory: (usize, usize),
    /// Range of output neurons, indexing into y
    pub action: (usize, usize),
}

impl Network for Continuous {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            self.y += (((&self.y + &self.θ).apply(&σ) * &self.w) - &self.y + &m_input)
                .elemul(&self.τ)
                .apply(&|v| v * inv);
        }
    }

    fn flush(&mut self) {
        self.y = Matrix::zeros(1, self.y.cols());
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}

impl Recurrent for Continuous {}

impl Stateful for Continuous {}

impl<C: Connection, G: Genome<C>> FromGenome<C, G> for Continuous {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.node_count();
        Self {
            y: Matrix::zeros(1, cols),
            θ: Matrix::zeros(1, cols),
            τ: Matrix::new(1, cols, vec![1.0; cols]),
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

        /// matrices have correct dimensions for node count
        #[test]
        fn test_matrix_dimensions() {
            let (genome, _) = G::new(3, 2);
            let nn = Continuous::from_genome(&genome);
            let cols = genome.node_count();

            assert_eq!(nn.y.cols(), cols);
            assert_eq!(nn.y.rows(), 1);
            assert_eq!(nn.w.cols(), cols);
            assert_eq!(nn.w.rows(), cols);
            assert_eq!(nn.θ.cols(), cols);
            assert_eq!(nn.τ.cols(), cols);
            assert_eq!(nn.τ.data()[0], 1.0);
        }

        /// sensory/action ranges map correctly to tuples
        #[test]
        fn test_bounds_mapped() {
            let (genome, _) = G::new(3, 2);
            let nn = Continuous::from_genome(&genome);

            assert_eq!(nn.sensory.0, genome.sensory().start);
            assert_eq!(nn.sensory.1, genome.sensory().end);
            assert_eq!(nn.action.0, genome.action().start);
            assert_eq!(nn.action.1, genome.action().end);
        }

        /// disabled connections excluded from weight matrix
        #[test]
        fn test_filters_disabled_connections() {
            let (mut genome, _) = G::new(2, 2);
            if let Some(c) = genome.connections_mut().first_mut() {
                c.disable();
            }

            let nn = Continuous::from_genome(&genome);
            let cols = genome.node_count();

            assert_eq!(nn.w.data()[0 * cols + 2], 0.0);
        }

        /// all disabled connections give zero weight matrix
        #[test]
        fn test_all_disabled_connections() {
            let (mut genome, _) = G::new(2, 2);
            for c in genome.connections_mut().iter_mut() {
                c.disable();
            }

            let nn = Continuous::from_genome(&genome);
            assert!(nn.w.data().iter().all(|&x| x == 0.0));
        }

        /// flush() zeroes state matrix
        #[test]
        fn test_flush_zeroes_state() {
            let (genome, _) = G::new(2, 2);
            let mut nn = Continuous::from_genome(&genome);
            let input = vec![1.0, 0.5];

            nn.step(5, &input, |x| x);
            assert!(nn.y.data().iter().any(|&x| x != 0.0));

            nn.flush();
            assert!(nn.y.data().iter().all(|&x| x == 0.0));
        }
    }
}
