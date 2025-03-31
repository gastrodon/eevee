pub mod recurrent;
pub mod serialize;
pub use recurrent::Ctrnn;

use crate::{Connection, Genome, Node};
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
}

pub trait Network: Serialize + for<'de> Deserialize<'de> {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F);
    fn flush(&mut self);
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

/// A network propagating non-linearly, where propagation through
/// recurrent connections is computed and invalid
pub trait Recurrent: Network {}

/// A network propagating linearly, where propagation through
/// recurrent connections won't be computed and may be invalid
pub trait Linear: Network {}

/// A network that retains state between calls to step,
/// where that state may interact with new input, or change output
pub trait Stateful: Network {}

/// A network that doesn't retain state between calls to step
pub trait Stateless: Network {}

pub trait FromGenome<N: Node, C: Connection<N>, G: Genome<N, C>>: Network {
    fn from_genome(genome: &G) -> Self;
}

pub trait ToNetwork<NN: Network, N: Node, C: Connection<N>>: Genome<N, C> {
    fn network(&self) -> NN;
}

impl<NN: Network, N: Node, C: Connection<N>, G: Genome<N, C>> ToNetwork<NN, N, C> for G
where
    NN: FromGenome<N, C, G>,
{
    fn network(&self) -> NN {
        NN::from_genome(self)
    }
}
