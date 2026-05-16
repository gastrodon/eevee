use super::{Connection, Genome, InnoGen};
use crate::crossover::crossover;
use core::cmp::Ordering;
use rand::{seq::IteratorRandom, RngCore};
use std::{collections::HashSet, fmt::Debug, marker::PhantomData};

/// Determines which `(from, to)` node pairs are valid new connections.
pub trait PathPolicy<C: Connection>: Clone + Debug + Default {
    fn allows(from: usize, to: usize, connections: &[C]) -> bool;
}

/// A neural-network-style genome parameterized by connection-topology policy `P`.
///
/// Node layout: `[0..sensory)` sensory, `[sensory..sensory+action)` action,
/// `(sensory+action..)` internal.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serialize",
    serde(bound(
        serialize = "C: Connection + serde::Serialize",
        deserialize = "C: Connection + for<'de2> serde::Deserialize<'de2>",
    ))
)]
#[derive(Debug, Clone)]
pub struct NNOrganism<C: Connection, P: PathPolicy<C>> {
    pub(crate) sensory: usize,
    pub(crate) action: usize,
    pub(crate) node_count: usize,
    pub(crate) connections: Vec<C>,
    #[cfg_attr(feature = "serialize", serde(skip))]
    _policy: PhantomData<P>,
}

impl<C: Connection, P: PathPolicy<C>> Genome<C> for NNOrganism<C, P> {
    fn new(sensory: usize, action: usize) -> (Self, usize) {
        let node_count = sensory + action;
        let mut inno = InnoGen::new(0);
        let mut connections = Vec::new();
        for from in 0..sensory {
            for to in sensory..node_count {
                connections.push(C::new(from, to, &mut inno));
            }
        }

        (
            Self {
                sensory,
                action,
                node_count,
                connections,
                _policy: PhantomData,
            },
            inno.head,
        )
    }

    fn sensory(&self) -> std::ops::Range<usize> {
        0..self.sensory
    }

    fn action(&self) -> std::ops::Range<usize> {
        self.sensory..self.sensory + self.action
    }

    fn node_count(&self) -> usize {
        self.node_count
    }

    fn push_node(&mut self) {
        self.node_count += 1;
    }

    fn connections(&self) -> &[C] {
        &self.connections
    }

    fn connections_mut(&mut self) -> &mut [C] {
        &mut self.connections
    }

    fn push_connection(&mut self, connection: C) {
        self.connections.push(connection);
    }

    fn open_path(&self, rng: &mut impl RngCore) -> Option<(usize, usize)> {
        let action_end = self.sensory + self.action;
        let mut saturated = HashSet::new();
        loop {
            // from: any non-action node (sensory or internal)
            let from = (0..self.node_count)
                .filter(|&i| !(i >= self.sensory && i < action_end) && !saturated.contains(&i))
                .choose(rng)?;

            let exclude: HashSet<usize> = self
                .connections
                .iter()
                .filter_map(|c| (c.from() == from).then_some(c.to()))
                .collect();

            // to: any non-sensory node (internal or action)
            if let Some(to) = (0..self.node_count)
                .filter(|&i| {
                    i >= self.sensory
                        && !exclude.contains(&i)
                        && P::allows(from, i, &self.connections)
                })
                .choose(rng)
            {
                break Some((from, to));
            }

            saturated.insert(from);
        }
    }

    fn reproduce_with(&self, other: &Self, self_fit: Ordering, rng: &mut impl RngCore) -> Self {
        let connections = crossover(&self.connections, &other.connections, self_fit, rng);
        let max_idx = connections
            .iter()
            .fold(0usize, |acc, c| acc.max(c.from()).max(c.to()));
        let node_count = (max_idx + 1).max(self.sensory + self.action);

        debug_assert!(
            connections
                .iter()
                .fold(0usize, |acc, c| acc.max(c.from()).max(c.to()))
                < node_count
        );

        Self {
            sensory: self.sensory,
            action: self.action,
            node_count,
            connections,
            _policy: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        genome::{connection::BWConnection, Genome, InnoGen, WConnection},
        random::default_rng,
    };
    use eevee_macros::fn_matrix;

    #[derive(Clone, Debug, Default)]
    struct AllowAll;
    impl<C: Connection> PathPolicy<C> for AllowAll {
        fn allows(_: usize, _: usize, _: &[C]) -> bool {
            true
        }
    }

    fn_matrix! {
        C: WConnection | BWConnection,
        G: NNOrganism<C, AllowAll>,

        /// basic genome creation test
        #[test]
        fn test_genome_creation() {
            let (genome, inno_head) = G::new(3, 2);
            assert_eq!(inno_head, 6);
            assert_eq!(genome.sensory().len(), 3);
            assert_eq!(genome.action().len(), 2);
            assert_eq!(genome.node_count(), 5);
            assert_eq!(genome.sensory, 3);
            assert_eq!(genome.action, 2);
            // layout: [0,1,2]=sensory [3,4]=action
            assert_eq!(genome.sensory + genome.action, 5);
        }

        /// empty genome
        #[test]
        fn test_genome_creation_empty() {
            let (genome, inno_head) = G::new(0, 0);
            assert_eq!(inno_head, 0);
            assert_eq!(genome.sensory().len(), 0);
            assert_eq!(genome.action().len(), 0);
            assert_eq!(genome.node_count(), 0);
            assert_eq!(genome.sensory + genome.action, 0);
        }

        /// only sensory nodes
        #[test]
        fn test_genome_creation_only_sensory() {
            let (genome, inno_head) = G::new(3, 0);
            assert_eq!(inno_head, 0);
            assert_eq!(genome.sensory().len(), 3);
            assert_eq!(genome.action().len(), 0);
            assert_eq!(genome.node_count(), 3);
            assert_eq!(genome.sensory, 3);
            // layout: [0,1,2]=sensory
            assert_eq!(genome.sensory + genome.action, 3);
        }

        /// only action nodes
        #[test]
        fn test_genome_creation_only_action() {
            let (genome, inno_head) = G::new(0, 3);
            assert_eq!(inno_head, 0);
            assert_eq!(genome.sensory().len(), 0);
            assert_eq!(genome.action().len(), 3);
            assert_eq!(genome.node_count(), 3);
            assert_eq!(genome.action, 3);
            // layout: [0,1,2]=action
            assert_eq!(genome.sensory + genome.action, 3);
        }

        /// bisection creates node and updates connections
        #[test]
        fn test_mutate_bisection() {
            let mut inno = InnoGen::new(0);
            let (mut genome, _) = G::new(1, 1);

            genome.connections = vec![];
            genome.push_connection({
                let mut c = C::new(0, 1, &mut inno);
                c.mutate_param(&mut default_rng());
                c
            });

            let innogen = &mut InnoGen::new(1);
            genome.bisect_connection(&mut default_rng(), innogen).unwrap_or_else(|e| panic!("failed bisect_connection: {e}"));

            assert!(!genome.connections()[0].enabled);

            assert_eq!(genome.connections()[1].from(), 0);
            assert_eq!(genome.connections()[1].to(), 2);
            assert_eq!(genome.connections()[1].weight(), 1.0);
            assert!(genome.connections()[1].enabled);
            assert_eq!(
                genome.connections()[1].inno,
                innogen.path((genome.connections()[1].from(), genome.connections()[1].to()))
            );

            assert_eq!(genome.connections()[2].from(), 2);
            assert_eq!(genome.connections()[2].to(), 1);
            assert_eq!(genome.connections()[1].weight(), 1.);
            assert_eq!(
                genome.connections()[2].weight(),
                genome.connections()[0].weight()
            );
            assert!(genome.connections()[2].enabled);
            assert_eq!(
                genome.connections()[2].inno,
                innogen.path((genome.connections()[2].from(), genome.connections()[2].to()))
            );

            assert_ne!(genome.connections()[0].inno, genome.connections()[1].inno);
            assert_ne!(genome.connections()[1].inno, genome.connections()[2].inno);
            assert_ne!(genome.connections()[0].inno, genome.connections()[2].inno);
        }

        /// empty genome cannot bisect
        #[test]
        fn test_mutate_bisection_empty_genome() {
            let (mut genome, _) = G::new(0, 0);
            genome.connections = vec![];
            assert!(genome.bisect_connection(&mut default_rng(), &mut InnoGen::new(0)).is_err());
        }

        /// no connections cannot bisect
        #[test]
        fn test_mutate_bisection_no_connections() {
            let (mut genome, _) = G::new(2, 2);
            genome.connections = vec![];
            assert!(genome.bisect_connection(&mut default_rng(), &mut InnoGen::new(0)).is_err());
        }
    }
}
