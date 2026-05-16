//! Traits and impls for Neural Networks derived from [Genome]s.
//!
//! Neural Networks who may be constructed from Genomes in order to
//! express their behaviour. The NEAT paper calls for a recurrent network with no per-connection
//! bias, though maybe we can do more than that here. The code inside is quite experimental.

pub mod continuous;
pub mod non_bias;
pub mod simple;

pub use continuous::Continuous;
pub use non_bias::NonBias;
pub use simple::Simple;

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

/// The trait for all networks. Right now, only f64 values are used.
pub trait Network {
    /// Given some sensory input, step the network with it `prec` times, activating with σ.
    /// Input must be sized to fit within [Genome::sensory].
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F);

    /// If the network is stateful, flush it's state
    fn flush(&mut self);

    /// Get the network's most recent output, which should be some range of neurons defined by
    /// [Genome::action].
    fn output(&self) -> &[f64];
}

/// Marker for a network who propagates non-linearly, where propagation through recurrent
/// connections is computed and valid.
pub trait Recurrent: Network {}

/// Marker for a network who propogates linearly, where propagation through recurrent
/// connections won't be computed and may be invalid
pub trait Linear: Network {}

/// Marker for a network who retains state between calls to step, where that state may interact
/// with new input, or change output
pub trait Stateful: Network {}

/// Marker for a network who doesn't retain state between calls to step
pub trait Stateless: Network {}

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
        NN: Continuous | NonBias,

        /// output size matches genome action neurons
        #[test]
        fn test_from_genome_output_bounds() {
            let (genome, _) = G::new(3, 2);
            let nn = NN::from_genome(&genome);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// zero action neurons produces empty output
        #[test]
        fn test_from_genome_empty_action() {
            let (genome, _) = G::new(3, 0);
            let nn = NN::from_genome(&genome);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// step() accepts sensory-sized input
        #[test]
        fn test_step_with_correct_input_size() {
            let (genome, _) = G::new(3, 2);
            let mut nn = NN::from_genome(&genome);
            let input: Vec<_> = (0..genome.sensory().len()).map(|i| i as f64).collect();
            nn.step(1, &input, |x| x);
            assert_eq!(nn.output().len(), genome.action().len());
        }

        /// flush() resets state; output changes on new steps
        #[test]
        fn test_flush_resets_state() {
            let (genome, _) = G::new(2, 2);
            let mut nn = NN::from_genome(&genome);
            let input = vec![1.0, 0.5];

            nn.step(3, &input, |x| x.signum());
            let output_before_flush = nn.output().to_vec();

            nn.flush();
            nn.step(1, &input, |x| x.signum());
            let output_after_flush = nn.output().to_vec();

            if !genome.connections().iter().any(|c| c.enabled()) {
                assert_eq!(output_before_flush, output_after_flush);
            } else {
                let magnitude_before: f64 = output_before_flush.iter().map(|x| x.abs()).sum();
                let magnitude_after: f64 = output_after_flush.iter().map(|x| x.abs()).sum();
                assert_ne!(magnitude_before, magnitude_after);
            }
        }

        /// different activation functions produce different outputs
        #[test]
        fn test_different_activations_differ() {
            let (genome, _) = G::new(2, 2);
            let mut nn1 = NN::from_genome(&genome);
            let mut nn2 = NN::from_genome(&genome);
            let input = vec![0.5, -0.5];

            nn1.step(3, &input, |x| x);
            nn2.step(3, &input, |x| x.abs());

            if genome.connections().iter().any(|c| c.enabled()) {
                let magnitude1: f64 = nn1.output().iter().map(|x| x.abs()).sum();
                let magnitude2: f64 = nn2.output().iter().map(|x| x.abs()).sum();
                assert_ne!(magnitude1, magnitude2);
            }
        }

        /// higher prec accumulates state differently
        #[test]
        fn test_prec_affects_convergence() {
            let (genome, _) = G::new(2, 2);
            let mut nn1 = NN::from_genome(&genome);
            let mut nn2 = NN::from_genome(&genome);
            let input = vec![1.0, 0.5];

            nn1.step(1, &input, |x| x.tanh());
            nn2.step(5, &input, |x| x.tanh());

            if genome.connections().iter().any(|c| c.enabled()) {
                assert_ne!(nn1.output(), nn2.output());
            }
        }
    }
}
