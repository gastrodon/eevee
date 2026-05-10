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
