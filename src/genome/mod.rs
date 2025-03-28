pub mod recurrent;
pub use recurrent::{CTRConnection, CTRGenome};

use crate::{random::Happens, specie::InnoGen, Network};
use core::{cmp::Ordering, error::Error, hash::Hash};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub trait Connection: Serialize + for<'de> Deserialize<'de> + Clone + Hash + PartialEq {
    /// gene innovation id
    fn inno(&self) -> usize;

    /// view and toggle whether or not the gene is enabled
    fn enabled(&self) -> bool;
    fn enable(&mut self);
    fn disable(&mut self);

    /// difference of connection parameters ( for example, weight )
    /// between this and another connection with the same innovation id
    fn param_diff(&self, other: &Self) -> f64;
}

pub trait Genome: Serialize + for<'de> Deserialize<'de> + Clone {
    type Connection: Connection;
    type Network: Network;

    /// A new genome of this type, with a known input and output size
    fn new(sensory: usize, action: usize) -> (Self, usize);

    /// A collection to the connections comprising this genome
    fn connections(&self) -> &[Self::Connection];

    /// Perform a ( possible? TODO ) mutation across every weight
    fn mutate_weights(&mut self, rng: &mut (impl RngCore + Happens));

    /// Generate a new connection
    fn mutate_connection(&mut self, rng: &mut (impl RngCore + Happens), inno: &mut InnoGen);

    /// Bisect an existing connection. Should panic if there are no connections to bisect
    fn mutate_bisection(&mut self, rng: &mut (impl RngCore + Happens), inno: &mut InnoGen);

    /// Perform 0 or more mutations on this genome ( should this be the only mutator exposed? TODO )
    fn maybe_mutate(&mut self, rng: &mut (impl RngCore + Happens), inno: &mut InnoGen);

    /// Perform crossover reproduction with other, where our fitness is `fitness_cmp` compared to other
    fn reproduce_with(
        &self,
        other: &Self,
        fitness_cmp: Ordering,
        rng: &mut (impl RngCore + Happens),
    ) -> Self;

    fn network(&self) -> Self::Network;

    fn to_string(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(self)?)
    }

    #[allow(clippy::should_implement_trait)]
    fn from_str(s: &str) -> Result<Self, Box<dyn Error>> {
        serde_json::from_str(s).map_err(|op| op.into())
    }

    fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        fs::write(path, self.to_string()?)?;
        Ok(())
    }

    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        Self::from_str(&fs::read_to_string(path)?)
    }
}
