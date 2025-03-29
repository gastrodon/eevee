pub mod recurrent;
pub use recurrent::{CTRConnection, CTRGenome};

use crate::{
    random::{EvolutionEvent, Happens},
    specie::InnoGen,
    Network,
};
use core::{cmp::Ordering, error::Error, fmt::Debug, hash::Hash};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub enum NodeKind {
    Sensory,
    Action,
    Internal,
    Bias,
}

pub trait Node: Serialize + for<'de> Deserialize<'de> + Clone + Debug {
    fn kind(&self) -> NodeKind;
    fn bias(&self) -> f64;
}

pub trait Connection:
    Serialize + for<'de> Deserialize<'de> + Clone + Hash + PartialEq + Default + Debug
{
    const EXCESS_COEFFICIENT: f64;
    const DISJOINT_COEFFICIENT: f64;
    const PARAM_COEFFICIENT: f64;

    type Node: Node;

    /// gene innovation id
    fn inno(&self) -> usize;

    /// whether or not this connection is active, and therefore affects its genomes behavior
    fn enabled(&self) -> bool;

    /// unconditionally enable this connection
    fn enable(&mut self);

    /// unconditionally disable this connection
    fn disable(&mut self);

    /// difference of connection parameters ( for example, weight )
    /// between this and another connection with the same innovation id
    fn param_diff(&self, other: &Self) -> f64;
}

pub trait Genome: Serialize + for<'de> Deserialize<'de> + Clone {
    type Node: Node = <<Self as Genome>::Connection as Connection>::Node;
    type Connection: Connection;
    type Network: Network;

    /// A new genome of this type, with a known input and output size
    fn new(sensory: usize, action: usize) -> (Self, usize);

    fn nodes(&self) -> &[Self::Node];

    /// Push a new node onto the genome
    fn push_node(&mut self, node: Self::Node);

    /// A collection to the connections comprising this genome
    fn connections(&self) -> &[Self::Connection];

    /// Mutable reference to the connections comprising this genome
    fn connections_mut(&mut self) -> &mut [Self::Connection];

    /// Push a connection onto the genome
    fn push_connection(&mut self, connection: Self::Connection);

    /// Push 2 connections onto the genome, first then second.
    /// The idea with this is that we'll often do so as a result of bisection,
    /// so this gives us a chance to grow the connections just once if we want
    fn push_2_connections(&mut self, first: Self::Connection, second: Self::Connection) {
        self.push_connection(first);
        self.push_connection(second);
    }

    /// Perform a ( possible? TODO ) mutation across every weight
    fn mutate_params(&mut self, rng: &mut (impl RngCore + Happens));

    /// Generate a new connection
    fn mutate_connection(&mut self, rng: &mut (impl RngCore + Happens), inno: &mut InnoGen);

    /// Bisect an existing connection. Should panic if there are no connections to bisect
    fn mutate_bisection(&mut self, rng: &mut (impl RngCore + Happens), inno: &mut InnoGen);

    /// Perform 0 or more mutations on this genome ( should this be the only mutator exposed? TODO )
    fn mutate(&mut self, rng: &mut (impl RngCore + Happens), innogen: &mut InnoGen) {
        if rng.happens(EvolutionEvent::MutateWeight) {
            self.mutate_params(rng);
        }
        if rng.happens(EvolutionEvent::MutateConnection) {
            self.mutate_connection(rng, innogen);
        }
        if rng.happens(EvolutionEvent::MutateBisection) && !self.connections().is_empty() {
            self.mutate_bisection(rng, innogen);
        }
    }

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
