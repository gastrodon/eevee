//! Neural Networks who may be constructed from [Genome]s in order to
//! express their behaviour. The NEAT paper calls for a recurrent network with no per-connection
//! bias, though maybe we can do more than that here. The code inside is quite experimental.

pub mod continuous;
pub mod non_bias;
pub mod simple;

pub use continuous::Continuous;
pub use non_bias::NonBias;
pub use simple::Simple;

use crate::{Connection, Genome};
use core::error::Error;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

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
pub trait Network: Serialize + for<'de> Deserialize<'de> {
    /// Given some sensory input, step the network with it `prec` times, activating with σ.
    /// Input must be sized to fit within [Genome::sensory].
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F);

    /// If the network is stateful, flush it's state
    fn flush(&mut self);

    /// Get the network's most recent output, which should be some range of neurons defined by
    /// [Genome::action].
    fn output(&self) -> &[f64];

    fn to_string(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(self)?)
    }

    fn from_str(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        serde_json::from_str(s).map_err(|op| op.into())
    }

    fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        fs::write(path, self.to_string()?)?;
        Ok(())
    }

    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        Self::from_str(&fs::read_to_string(path)?)
    }
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
