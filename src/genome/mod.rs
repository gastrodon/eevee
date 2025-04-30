//! Traits and impls for Genomes and their Connections.
//!
//! A [Genome] descirbes some discrete structure which may be mutated, evaluated, and be
//! comprised of a collection of connectionss who describe some network structure. Genomes may
//! ( but not necessarily ) be marshallable into a [Neural Network](crate::network::Network).
//!
//! A [Connection] describes a member of a genome's body ( sometimes referred to as a gene )
//! that describes some discrete behavior. In aggregate, connections may describe arbitrarially
//! complex behavior. Through evolution, that complex behavior is refined towards increasing
//! some one-dimensional fitness.
pub mod connection;
pub mod recurrent;

pub use connection::WConnection;
pub use recurrent::Recurrent;

use crate::random::{percent, ConnectionEvent, EventKind, GenomeEvent};
use core::{cmp::Ordering, error::Error, fmt::Debug, hash::Hash, ops::Range};
use fxhash::FxHashMap;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

/// InnoGen is a structure who's job is to associate an innovation ID uniquely with some
/// connection path in the from (from, to). It typically lives generationally, ie every new
/// connection through some path formed in a single generation should have the same innovation
/// id as every other connection through the same path formed that generation so that they can
/// later be used in crossover reproduction.
pub struct InnoGen {
    pub head: usize,
    seen: FxHashMap<(usize, usize), usize>,
}

impl InnoGen {
    pub fn new(head: usize) -> Self {
        Self {
            head,
            seen: FxHashMap::default(),
        }
    }

    pub fn path(&mut self, v: (usize, usize)) -> usize {
        match self.seen.get(&v) {
            Some(n) => *n,
            None => {
                let n = self.head;
                self.head += 1;
                self.seen.insert(v, n);
                n
            }
        }
    }
}

/// This has no reason to exist, and will be replaced with ranges in the future.
#[deprecated]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Sensory,
    Action,
    Internal,
    Static,
}

#[derive(Debug, Clone, Copy)]
pub enum ConnectionPoint {
    From(usize),
    To(usize),
}

/// A connection between 2 points. Connections may be arbitrarially parameterized, and those
/// parameters mutated inside [mutate_param](Connection::mutate_param). For those params to
/// actually be _used_, a connection should expose them with a trait, and a
/// [Network](crate::network::Network) implementer should know about them. Any connection must
/// have a path, weight, and innovation_id ( which should be supplied from InnoGen ).
pub trait Connection:
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

    /// path destination
    fn to(&self) -> usize {
        self.path().1
    }

    fn weight(&self) -> f64;

    /// difference of connection parameters ( for example, weight )
    /// between this and another connection with the same innovation id
    fn param_diff(&self, other: &Self) -> f64;

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
}

/// A genome comprised of some connections and connections. A genome must be able to form new
/// new connections, bisect any existing connection, and mutate any existing connections
/// arbitrary parameters. A genome must also be able to reproduce with any other genome of the
/// same kind, their connections constructively crossing over.
pub trait Genome<C: Connection>: Serialize + for<'de> Deserialize<'de> + Clone {
    const MUTATE_NODE_PROBABILITY: u64 = percent(20);
    const MUTATE_CONNECTION_PROBABILITY: u64 = percent(20);
    const PROBABILITIES: [u64; GenomeEvent::COUNT] =
        [percent(5), percent(15), percent(80), percent(0)];

    /// A new genome of this type, with a known input and output size.
    fn new(sensory: usize, action: usize) -> (Self, usize);

    fn sensory(&self) -> Range<usize>;

    fn action(&self) -> Range<usize>;

    fn nodes(&self) -> &[NodeKind];

    #[deprecated]
    fn nodes_mut(&mut self) -> &mut [NodeKind];

    /// Push a new node onto the genome.
    fn push_node(&mut self, node: NodeKind);

    /// A collection to the connections comprising this genome.
    fn connections(&self) -> &[C];

    /// Mutable reference to the connections comprising this genome.
    fn connections_mut(&mut self) -> &mut [C];

    /// Push a connection onto the genome.
    fn push_connection(&mut self, connection: C);

    /// Push 2 connections onto the genome, first then second.
    /// The idea with this is that we'll often do so as a result of bisection, so this gives us
    /// a chance to grow the connections just once if we want.
    fn push_2_connections(&mut self, first: C, second: C) {
        self.push_connection(first);
        self.push_connection(second);
    }

    /// Possibly mutate a single connection. On average, will mutate every
    /// [MUTATE_CONNECTION_PROBABILITY](Genome::MUTATE_CONNECTION_PROBABILITY) / [u64::MAX]
    /// connection.
    fn mutate_connection(&mut self, rng: &mut impl RngCore) {
        for c in self.connections_mut() {
            if rng.next_u64() < Self::MUTATE_CONNECTION_PROBABILITY {
                c.mutate(rng);
            }
        }
    }

    /// Find some open path ( that is, a path between nodes from -> to ) that no connection is
    /// occupying if any exist. Whatever path is returned will be considered valid, and may be
    /// used when generating a new connection.
    /// `included` describes a part of the path being queried, if any. If Some(From(n)),
    /// the path must originate from n, if Some(To(n)) the path must travel to n.
    fn open_path(
        &self,
        included: Option<ConnectionPoint>,
        rng: &mut impl RngCore,
    ) -> Option<(usize, usize)>;

    /// Generate a new connection between unconnected nodes. Panics if all possible connections
    /// between nodes are saturated ( TODO: is that a good idea? )
    fn new_connection(&mut self, rng: &mut impl RngCore, inno: &mut InnoGen) {
        if let Some((from, to)) = self.open_path(None, rng) {
            self.push_connection(C::new(from, to, inno));
        } else {
            panic!("connections on genome are fully saturated")
        }
    }

    /// Bisect an existing connection. Panics if there are no connections to bisect. This is the
    /// mechanism by which the internal / "hidden" layer of nodes grows on a genome, the new
    /// node being at the center of the bisection.
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

        self.push_node(NodeKind::Internal);
        self.push_2_connections(lower, upper);
    }

    /// Perform 0 or more mutations on this genome. If [PROBABILITIES](Genome::PROBABILITIES)
    /// add up to [u64::MAX], some event will always be picked. Otherwise, it's possible that
    /// no mutation actually ocurrs.
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
                GenomeEvent::MutateNode => unreachable!("nodes may not be mutated"),
            }
        }
    }

    /// Perform crossover reproduction with other, where our fitness is `fitness_cmp` compared to other
    fn reproduce_with(&self, other: &Self, fitness_cmp: Ordering, rng: &mut impl RngCore) -> Self;

    /// Serialize this genome to a JSON string
    fn to_string(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(self)?)
    }

    /// Deserialize this genome from a JSON string
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
