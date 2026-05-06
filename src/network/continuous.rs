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
        let static_idx = genome.action().end;
        Self {
            y: Matrix::zeros(1, cols),
            θ: Matrix::new(
                1,
                cols,
                (0..cols)
                    .map(|i| if i == static_idx { 1. } else { 0. })
                    .collect::<Vec<_>>(),
            ),
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
    use crate::{
        assert_f64_approx,
        genome::{self, InnoGen, WConnection},
    };

    #[test]
    fn test_from_genome() {
        type C = WConnection;

        let mut inno = InnoGen::new(0);
        let (mut genome, _) = genome::Recurrent::<C>::new(2, 2);
        genome.push_connection(C::new(0, 3, &mut inno));
        genome.push_connection(C::new(0, 1, &mut inno));
        genome.push_connection(C::new(0, 1, &mut inno));

        let nn = Continuous::from_genome(&genome);
        unsafe {
            for c in genome.connections() {
                if c.enabled() {
                    assert_f64_approx!(nn.w.get_unchecked([c.from(), c.to()]), c.weight());
                }
            }

            let static_idx = genome.action().end;
            for i in 0..genome.node_count() {
                assert_f64_approx!(
                    nn.θ.get_unchecked([0, i]),
                    if i == static_idx { 1. } else { 0. }
                )
            }
        }

        assert_eq!((nn.sensory.0, nn.sensory.1), (genome.sensory().start, genome.sensory().end));
        assert_eq!((nn.action.0, nn.action.1), (genome.action().start, genome.action().end));
    }
}
