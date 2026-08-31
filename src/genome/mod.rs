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
pub mod nn_organism;
pub mod nn_policies;

pub use connection::WConnection;
pub use nn_organism::{NNOrganism, PathPolicy};
pub use nn_policies::{NonRecurrent, Recurrent};

use crate::random::{percent, ConnectionEvent, EventKind, GenomeEvent};
use core::{cmp::Ordering, error::Error, fmt::Debug, hash::Hash, ops::Range};
use fxhash::FxHashMap;
use rand::{Rng, RngCore};

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

/// A connection between 2 points. Connections may be arbitrarially parameterized, and those
/// parameters mutated inside [mutate_param](Connection::mutate_param). For those params to
/// actually be _used_, a connection should expose them with a trait, and a
/// [Network](crate::network::Network) implementer should know about them. Any connection must
/// have a path, weight, and innovation_id ( which should be supplied from InnoGen ).
pub trait Connection: Clone + Hash + PartialEq + Default + Debug {
    const PROBABILITIES: [u64; ConnectionEvent::COUNT] = [percent(1), percent(99)];
    const PARAM_REPLACE_PROBABILITY: u64 = percent(20);
    const PARAM_PERTURB_FAC: f64 = 0.45;
    const PARAM_STD: f64 = 3.;

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
pub trait Genome<C: Connection>: Clone {
    const MUTATE_NODE_PROBABILITY: u64 = percent(20);
    const MUTATE_CONNECTION_PROBABILITY: u64 = percent(30);
    const PROBABILITIES: [u64; GenomeEvent::COUNT] =
        [percent(10), percent(10), percent(80), percent(0)];

    /// A new genome of this type, with a known input and output size.
    fn new(sensory: usize, action: usize) -> (Self, usize);

    fn sensory(&self) -> Range<usize>;

    fn action(&self) -> Range<usize>;

    /// Total number of nodes. Layout: `[0..sensory)` sensory, `[sensory..+action)` action,
    /// `(sensory+action..)` internal.
    fn node_count(&self) -> usize;

    /// Push a new internal node.
    fn push_node(&mut self);

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
    fn open_path(&self, rng: &mut impl RngCore) -> Option<(usize, usize)>;

    /// Generate a new connection between unconnected nodes. Fails if all possible connections
    /// between nodes are saturated
    fn new_connection(
        &mut self,
        rng: &mut impl RngCore,
        inno: &mut InnoGen,
    ) -> Result<(), Box<dyn Error>> {
        if let Some((from, to)) = self.open_path(rng) {
            self.push_connection(C::new(from, to, inno));
            Ok(())
        } else {
            Err("connections on genome are fully saturated".into())
        }
    }

    /// Bisect an existing connection. Fails if there are no connections to bisect. This is the
    /// mechanism by which the internal / "hidden" layer of nodes grows on a genome, the new
    /// node being at the center of the bisection.
    fn bisect_connection(
        &mut self,
        rng: &mut impl RngCore,
        inno: &mut InnoGen,
    ) -> Result<(), Box<dyn Error>> {
        if self.connections().is_empty() {
            return Err("no connections available to bisect".into());
        }

        let center = self.node_count();
        let source = rng.random_range(0..self.connections().len());
        let (lower, upper) = self
            .connections_mut()
            .get_mut(source)
            .unwrap()
            .bisect(center, inno);

        self.push_node();
        self.push_2_connections(lower, upper);
        Ok(())
    }

    /// Perform 0 or more mutations on this genome. If [PROBABILITIES](Genome::PROBABILITIES)
    /// add up to [u64::MAX], some event will always be picked. Otherwise, it's possible that
    /// no mutation actually ocurrs.
    fn mutate(
        &mut self,
        rng: &mut impl RngCore,
        innogen: &mut InnoGen,
    ) -> Result<(), Box<dyn Error>> {
        if self.connections().is_empty() {
            self.new_connection(rng, innogen)?;
        } else if let Some(evt) = GenomeEvent::pick(rng, Self::PROBABILITIES) {
            match evt {
                GenomeEvent::NewConnection => match self.open_path(rng) {
                    Some((from, to)) => self.push_connection(C::new(from, to, innogen)),
                    None => self.bisect_connection(rng, innogen)?,
                },
                GenomeEvent::BisectConnection => self.bisect_connection(rng, innogen)?,
                GenomeEvent::MutateConnection => self.mutate_connection(rng),
                GenomeEvent::MutateNode => unreachable!("nodes may not be mutated"),
            };
        }

        Ok(())
    }

    /// Perform crossover reproduction with other, where our fitness is `fitness_cmp` compared to other
    fn reproduce_with(&self, other: &Self, fitness_cmp: Ordering, rng: &mut impl RngCore) -> Self;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::{connection::BWConnection, NonRecurrent, Recurrent, WConnection};
    use crate::random::default_rng;
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,
        G: Recurrent<C> | NonRecurrent<C>,

        /// A freshly built genome with no internal nodes is fully saturated:
        /// every sensory→action path exists and nothing may point back into a
        /// sensory node. Mutation must still succeed — it can grow a node —
        /// rather than erroring and taking the whole reproduction down with it.
        #[test]
        fn test_mutate_saturated() {
            let mut rng = default_rng();
            for (sensory, action) in [(2, 1), (1, 1), (3, 1), (2, 2)] {
                let (genome, inno_head) = G::new(sensory, action);
                assert!(
                    genome.open_path(&mut rng).is_none(),
                    "{sensory}x{action} should start saturated"
                );

                // Force the NewConnection branch many times over; each must grow
                // the genome rather than return Err.
                for _ in 0..64 {
                    let mut g = genome.clone();
                    let mut inno = InnoGen::new(inno_head);
                    let before = g.node_count();
                    g.mutate(&mut rng, &mut inno)
                        .unwrap_or_else(|e| panic!("{sensory}x{action} mutate failed: {e}"));
                    assert!(g.node_count() >= before);
                }
            }
        }
    }
}
