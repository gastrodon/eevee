//! Traits and impls for Neural Networks derived from [Genome]s.
//!
//! Neural Networks who may be constructed from Genomes in order to
//! express their behaviour. The NEAT paper calls for a recurrent network with no per-connection
//! bias, though maybe we can do more than that here. The code inside is quite experimental.

pub mod binary;
pub mod feedforward;
pub mod realtime;

pub use binary::BinaryFeedForward;
pub use feedforward::FeedForward;
pub use realtime::{Realtime, RealtimeUnbias};

use crate::{Connection, Genome};

pub mod activate {
    use core::f64::consts::E;

    pub fn steep_sigmoid(x: f64) -> f64 {
        1. / (1. + E.powf(-4.9 * x))
    }

    pub fn relu(x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            x
        }
    }
}

pub mod loss {
    pub fn decay_quadratic(want: f64, x: f64) -> f64 {
        1. - (want - x).abs().powf(2.)
    }

    pub fn decay_linear(want: f64, have: f64) -> f64 {
        if have.is_nan() {
            f64::MIN
        } else {
            want - (want - have).abs()
        }
    }
}

/// Default integration steps for [Realtime] networks.
pub(crate) fn default_prec() -> usize {
    20
}

/// The trait for all networks. Right now, only f64 values are used.
pub trait Network {
    /// Evaluate the network given sensory input, activating with σ.
    /// Input must be sized to fit within [Genome::sensory].
    /// [Continuous] networks use their stored `prec` for integration steps.
    fn step<F: Fn(f64) -> f64>(&mut self, input: &[f64], σ: F);

    /// Get the network's most recent output, which should be some range of neurons defined by
    /// [Genome::action].
    fn output(&self) -> &[f64];
}

/// Marker for a network whose state persists across [Network::step] calls.
pub trait Continuous: Network {
    /// Reset the network's internal state.
    fn reset(&mut self);
}

/// Marker for a network with no state carried between [Network::step] calls.
pub trait Discrete: Network {}

/// For some [Genome], a network may construct itself from it.
pub trait FromGenome<C: Connection, G: Genome<C>>: Network {
    fn from_genome(genome: &G) -> Self;
}

/// The inverse of [FromGenome], implemented automatically by any [Network] for every
/// [Genome] from whom it knows how to construct itself.
pub trait ToNetwork<NN: Network, C: Connection>: Genome<C> {
    fn network(&self) -> NN;
}

impl<NN: Network, C: Connection, G: Genome<C>> ToNetwork<NN, C> for G
where
    NN: FromGenome<C, G>,
{
    fn network(&self) -> NN {
        NN::from_genome(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::{self, connection::BWConnection, WConnection};
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,
        G: genome::Recurrent<C>,
        NN: Realtime,

        /// output size matches genome action neurons
        #[test]
        fn test_realtime_output_bounds() {
            let (genome, _) = G::new(3, 2);
            let nn = NN::from_genome(&genome);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// zero action neurons produces empty output
        #[test]
        fn test_realtime_empty_action() {
            let (genome, _) = G::new(3, 0);
            let nn = NN::from_genome(&genome);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// step() accepts sensory-sized input
        #[test]
        fn test_realtime_step_accepts_input() {
            let (genome, _) = G::new(3, 2);
            let mut nn = NN::from_genome(&genome);
            let input: Vec<_> = (0..genome.sensory().len()).map(|i| i as f64).collect();
            nn.step(&input, |x| x);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// reset() resets internal state
        #[test]
        fn test_realtime_reset() {
            let (genome, _) = G::new(2, 2);
            let mut nn = NN::from_genome(&genome);
            let input = vec![1.0, 0.5];

            nn.step(&input, |x| x.signum());
            let _output_before = nn.output().to_vec();

            nn.reset();
            // After reset, state is cleared. Stepping again should produce output.
            nn.step(&input, |x| x.signum());
            let _output_after = nn.output().to_vec();
        }
    }

    fn_matrix! {
        C: WConnection | BWConnection,
        G: genome::NonRecurrent<C>,
        NN: FeedForward | BinaryFeedForward,

        /// output size matches genome action neurons
        #[test]
        fn test_feedforward_output_bounds() {
            let (genome, _) = G::new(3, 2);
            let nn = NN::from_genome(&genome);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// zero action neurons produces empty output
        #[test]
        fn test_feedforward_empty_action() {
            let (genome, _) = G::new(3, 0);
            let nn = NN::from_genome(&genome);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// step() accepts sensory-sized input
        #[test]
        fn test_feedforward_step_accepts_input() {
            let (genome, _) = G::new(3, 2);
            let mut nn = NN::from_genome(&genome);
            let input: Vec<_> = (0..genome.sensory().len()).map(|i| i as f64).collect();
            nn.step(&input, |x| x);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// multiple steps produce consistent outputs for stateless networks
        #[test]
        fn test_feedforward_consistent_output() {
            let (genome, _) = G::new(2, 2);
            let mut nn = NN::from_genome(&genome);
            let input = vec![1.0, 0.5];

            nn.step(&input, |x| x.signum());
            let output_first = nn.output().to_vec();

            nn.step(&input, |x| x.signum());
            let output_second = nn.output().to_vec();

            assert_eq!(output_first, output_second);
        }

        /// different activation functions can be applied
        #[test]
        fn test_feedforward_different_activations() {
            let (genome, _) = G::new(2, 2);
            let mut nn1 = NN::from_genome(&genome);
            let mut nn2 = NN::from_genome(&genome);
            let input = vec![0.5, -0.5];

            nn1.step(&input, |x| x);
            let _output1 = nn1.output().to_vec();

            nn2.step(&input, |x| x.abs());
            let _output2 = nn2.output().to_vec();
        }
    }
}
