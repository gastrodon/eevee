use super::{default_prec, Continuous, FromGenome};
use crate::{Connection, Genome, Network, Recurrent};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

/// A stateful recurrent NN for realtime / continuous-time problems.
///
/// Based on the CTRNN dynamics described in:
/// "On the dynamics of small continuous-time recurrent neural networks" (Beer 1995).
#[derive(Debug, Clone)]
pub struct Realtime {
    /// Integration steps per [Network::step] call.
    pub prec: usize,
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

impl Network for Realtime {
    fn step<F: Fn(f64) -> f64>(&mut self, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (self.prec as f64);
        for _ in 0..self.prec {
            self.y += (((&self.y + &self.θ).apply(&σ) * &self.w) - &self.y + &m_input)
                .elemul(&self.τ)
                .apply(&|v| v * inv);
        }
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}

impl crate::Recurrent for Realtime {}

impl Continuous for Realtime {
    fn reset(&mut self) {
        self.y = Matrix::zeros(1, self.y.cols());
    }
}

impl<C: Connection, G: Genome<C> + Recurrent> FromGenome<C, G> for Realtime {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.node_count();
        Self {
            prec: default_prec(),
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

#[derive(Debug)]
pub struct RealtimeUnbias {
    /// Integration steps per [Network::step] call.
    pub prec: usize,
    pub y: Matrix<f64>,
    pub w: Matrix<f64>,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl Network for RealtimeUnbias {
    fn step<F: Fn(f64) -> f64>(&mut self, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (self.prec as f64);
        for _ in 0..self.prec {
            self.y = ((&self.y + &m_input).apply(&σ) * &self.w).apply(&|v| v * inv);
        }
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}

impl Recurrent for RealtimeUnbias {}

impl Continuous for RealtimeUnbias {
    fn reset(&mut self) {
        self.y = Matrix::zeros(1, self.y.cols());
    }
}

impl<C: Connection, G: Genome<C> + Recurrent> FromGenome<C, G> for RealtimeUnbias {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.node_count();
        Self {
            prec: default_prec(),
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

        let nn = Realtime::from_genome(&genome);
        unsafe {
            for c in genome.connections() {
                if c.enabled() {
                    assert_f64_approx!(nn.w.get_unchecked([c.from(), c.to()]), c.weight());
                }
            }

            for i in 0..genome.node_count() {
                assert_f64_approx!(nn.θ.get_unchecked([0, i]), 0.)
            }
        }

        assert_eq!(
            (nn.sensory.0, nn.sensory.1),
            (genome.sensory().start, genome.sensory().end)
        );
        assert_eq!(
            (nn.action.0, nn.action.1),
            (genome.action().start, genome.action().end)
        );
    }
}
