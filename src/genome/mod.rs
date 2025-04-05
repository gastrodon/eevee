pub mod connection;
pub mod node;
pub mod recurrent;

pub use connection::WConnection;
pub use recurrent::CTRGenome;

use crate::{
    random::{percent, ConnectionEvent, EventKind, GenomeEvent},
    specie::InnoGen,
};
use core::{cmp::Ordering, error::Error, fmt::Debug, hash::Hash, ops::Range};
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Sensory,
    Action,
    Internal,
    Static,
}

pub trait Node: Serialize + for<'de> Deserialize<'de> + Clone + Debug + PartialEq {
    const PARAM_REPLACE_PROBABILITY: u64 = percent(10);
    const PARAM_PERTURB_FAC: f64 = 0.05;

    /// A new node of some kind
    fn new(kind: NodeKind) -> Self;

    /// The cond of node this is, where
    /// - sensory reads input to a network
    /// - action describes output from a network
    /// - internal is a hidden node of a network
    /// - static is a hidden node, but unaffected by input, yielding a static value
    fn kind(&self) -> NodeKind;

    fn mutate_param(&mut self, rng: &mut impl RngCore);

    fn mutate(&mut self, rng: &mut impl RngCore) {
        // this is redundant, but done to retain scaffolding for future non-param mutations
        self.mutate_param(rng);
    }
}

pub trait Biased: Node {
    fn bias(&self) -> f64;
}

pub trait Timescaled: Node {
    fn timescale(&self) -> f64;
}

pub trait Connection<N: Node>:
    Serialize + for<'de> Deserialize<'de> + Clone + Hash + PartialEq + Default + Debug
{
    const PROBABILITIES: [u64; ConnectionEvent::COUNT] = [percent(1), percent(99)];
    const PARAM_REPLACE_PROBABILITY: u64 = percent(10);
    const PARAM_PERTURB_FAC: f64 = 0.05;

    const EXCESS_COEFFICIENT: f64;
    const DISJOINT_COEFFICIENT: f64;
    const PARAM_COEFFICIENT: f64;

    const PROBABILITY_PICK_RL: u64 = percent(50);
    const PROBABILITY_KEEP_DISABLED: u64 = percent(75);

    fn new(from: usize, to: usize, inno: &mut InnoGen) -> Self;

    /// gene innovation id
    fn inno(&self) -> usize;

    /// whether or not this connection is active, and therefore affects its genomes behavior
    fn enabled(&self) -> bool;

    /// unconditionally enable this connection
    fn enable(&mut self);

    /// unconditionally disable this connection
    fn disable(&mut self);

    /// (from, to) path of this connection
    fn path(&self) -> (usize, usize);

    /// path source
    fn from(&self) -> usize {
        self.path().0
    }

    fn weight(&self) -> f64;

    /// path destination
    fn to(&self) -> usize {
        self.path().1
    }

    /// possibly mutate a single param
    fn mutate_param(&mut self, rng: &mut impl RngCore);

    /// mutate a connection
    fn mutate(&mut self, rng: &mut impl RngCore) {
        if let Some(evt) = ConnectionEvent::pick(rng, Self::PROBABILITIES) {
            match evt {
                ConnectionEvent::Disable => self.disable(),
                ConnectionEvent::MutateParam => self.mutate_param(rng),
            }
        }
    }

    /// bisect this connection; disabling it, and returning the (upper, lower) bisection pair
    fn bisect(&mut self, center: usize, inno: &mut InnoGen) -> (Self, Self);

    /// difference of connection parameters ( for example, weight )
    /// between this and another connection with the same innovation id
    fn param_diff(&self, other: &Self) -> f64;
}
pub trait Genome<N: Node, C: Connection<N>>: Serialize + for<'de> Deserialize<'de> + Clone {
    const PROBABILITIES: [u64; GenomeEvent::COUNT] =
        [percent(5), percent(15), percent(60), percent(20)];

    /// A new genome of this type, with a known input and output size
    fn new(sensory: usize, action: usize) -> (Self, usize);

    fn sensory(&self) -> Range<usize>;

    fn action(&self) -> Range<usize>;

    fn nodes(&self) -> &[N];

    fn nodes_mut(&mut self) -> &mut [N];

    /// Push a new node onto the genome
    fn push_node(&mut self, node: N);

    fn mutate_node(&mut self, rng: &mut impl RngCore) {
        let pick = rng.random_range(0..self.nodes().len());
        self.nodes_mut()[pick].mutate(rng);
    }

    /// A collection to the connections comprising this genome
    fn connections(&self) -> &[C];

    /// Mutable reference to the connections comprising this genome
    fn connections_mut(&mut self) -> &mut [C];

    /// Push a connection onto the genome
    fn push_connection(&mut self, connection: C);

    /// Push 2 connections onto the genome, first then second.
    /// The idea with this is that we'll often do so as a result of bisection,
    /// so this gives us a chance to grow the connections just once if we want
    fn push_2_connections(&mut self, first: C, second: C) {
        self.push_connection(first);
        self.push_connection(second);
    }

    /// Mutate a single connection
    fn mutate_connection(&mut self, rng: &mut impl RngCore) {
        let pick = rng.random_range(0..self.connections().len());
        self.connections_mut()[pick].mutate(rng);
    }

    /// Find some open path ( that is, a path between nodes from -> to )
    /// that no connection is occupying if any exist
    fn open_path(&self, rng: &mut impl RngCore) -> Option<(usize, usize)>;

    /// Generate a new connection between unconnected nodes.
    /// Panics if all possible connections between nodes are saturated
    fn new_connection(&mut self, rng: &mut impl RngCore, inno: &mut InnoGen) {
        if let Some((from, to)) = self.open_path(rng) {
            self.push_connection(C::new(from, to, inno));
        } else {
            panic!("connections on genome are fully saturated")
        }
    }

    /// Bisect an existing connection. Should panic if there are no connections to bisect
    fn bisect_connection(&mut self, rng: &mut impl RngCore, inno: &mut InnoGen) {
        if self.connections().is_empty() {
            panic!("no connections available to bisect");
        }

        let center = self.nodes().len();
        let source = rng.random_range(0..self.connections().len());
        let (lower, upper) = self
            .connections_mut()
            .get_mut(source)
            .unwrap()
            .bisect(center, inno);

        self.push_node(N::new(NodeKind::Internal));
        self.push_2_connections(lower, upper);
    }

    /// Perform 0 or more mutations on this genome ( should this be the only mutator exposed? TODO )
    fn mutate(&mut self, rng: &mut impl RngCore, innogen: &mut InnoGen) {
        if let Some(evt) = GenomeEvent::pick(rng, Self::PROBABILITIES) {
            match evt {
                GenomeEvent::NewConnection => self.new_connection(rng, innogen),
                GenomeEvent::BisectConnection => {
                    if !self.connections().is_empty() {
                        self.bisect_connection(rng, innogen)
                    }
                }
                GenomeEvent::MutateConnection => {
                    if !self.connections().is_empty() {
                        self.mutate_connection(rng)
                    }
                }
                GenomeEvent::MutateNode => self.mutate_node(rng),
            }
        }
    }

    /// Perform crossover reproduction with other, where our fitness is `fitness_cmp` compared to other
    fn reproduce_with(&self, other: &Self, fitness_cmp: Ordering, rng: &mut impl RngCore) -> Self;

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
