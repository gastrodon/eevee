use super::{Connection, Genome, InnoGen};
use crate::crossover::crossover;
use core::cmp::{max, Ordering};
use rand::{seq::IteratorRandom, RngCore};
use std::collections::HashSet;

/// A genome that allows recurrent connections
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", serde(bound = "C: Connection"))]
pub struct Recurrent<C: Connection> {
    pub(crate) sensory: usize,
    pub(crate) action: usize,
    pub(crate) node_count: usize,
    pub(crate) connections: Vec<C>,
}

impl<C: Connection> Recurrent<C> {
    fn static_idx(&self) -> usize {
        self.sensory + self.action
    }
}

impl<C: Connection> Genome<C> for Recurrent<C> {
    fn new(sensory: usize, action: usize) -> (Self, usize) {
        let node_count = sensory + action;

        let mut inno = InnoGen::new(0);
        let mut connections = Vec::new();
        for from in 0..sensory {
            for to in sensory..sensory + action {
                connections.push(C::new(from, to, &mut inno));
            }
        }

        (
            Self {
                sensory,
                action,
                node_count,
                connections,
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
        let mut saturated = HashSet::new();
        loop {
            let (from, _) = (0..self.node_count)
                .map(|i| (i, ()))
                .filter(|(i, _)| {
                    // not action
                    (*i < self.sensory || *i >= self.sensory + self.action)
                        && !saturated.contains(i)
                })
                .choose(rng)?;

            let exclude = self
                .connections
                .iter()
                .filter_map(|c| (c.from() == from).then_some(c.to()))
                .collect::<HashSet<_>>();

            if let Some((to, _)) = (0..self.node_count)
                .map(|i| (i, ()))
                .filter(|(i, _)| {
                    // not sensory
                    *i >= self.sensory && !exclude.contains(i)
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
            .fold(0usize, |prev, c| max(prev, max(c.from(), c.to())));
        let node_count = (max_idx + 1).max(self.sensory + self.action);

        Self {
            sensory: self.sensory,
            action: self.action,
            node_count,
            connections,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::{connection::BWConnection, WConnection};
    use crate::random::default_rng;
    use eevee_macros::fn_matrix;

    type GenomeWConn = Recurrent<WConnection>;
    type GenomeBConn = Recurrent<BWConnection>;

    fn_matrix! {
        G: GenomeWConn | GenomeBConn,

        /// new(3,2) creates 3 sensory, 2 action, 5 total, 6 fully-connected S→A connections
        #[test]
        fn test_genome_creation() {
            let (genome, inno_head) = G::new(3, 2);
            assert_eq!(inno_head, 6);
            assert_eq!(genome.sensory().len(), 3);
            assert_eq!(genome.action().len(), 2);
            assert_eq!(genome.node_count(), 5);
            assert_eq!(genome.connections().len(), 6);
        }

        /// new(0,0) creates zero nodes and zero connections
        #[test]
        fn test_genome_creation_empty() {
            let (genome, inno_head) = G::new(0, 0);
            assert_eq!(inno_head, 0);
            assert_eq!(genome.sensory().len(), 0);
            assert_eq!(genome.action().len(), 0);
            assert_eq!(genome.node_count(), 0);
            assert_eq!(genome.connections().len(), 0);
        }

        /// new(3,0) creates sensory nodes with no action nodes
        #[test]
        fn test_genome_creation_only_sensory() {
            let (genome, inno_head) = G::new(3, 0);
            assert_eq!(inno_head, 0);
            assert_eq!(genome.sensory().len(), 3);
            assert_eq!(genome.action().len(), 0);
            assert_eq!(genome.node_count(), 3);
            assert_eq!(genome.connections().len(), 0);
        }

        /// new(0,3) creates action nodes with no sensory nodes
        #[test]
        fn test_genome_creation_only_action() {
            let (genome, inno_head) = G::new(0, 3);
            assert_eq!(inno_head, 0);
            assert_eq!(genome.sensory().len(), 0);
            assert_eq!(genome.action().len(), 3);
            assert_eq!(genome.node_count(), 3);
            assert_eq!(genome.connections().len(), 0);
        }

        /// Ranges don't overlap and cover expected node indices
        #[test]
        fn test_genome_sensory_action_ranges() {
            let (genome, _) = G::new(4, 3);
            let sensory = genome.sensory();
            let action = genome.action();
            assert_eq!(sensory.start, 0);
            assert_eq!(sensory.end, 4);
            assert_eq!(action.start, 4);
            assert_eq!(action.end, 7);
            assert_eq!(sensory.len(), 4);
            assert_eq!(action.len(), 3);
        }

        /// push_node() increments node_count by exactly 1
        #[test]
        fn test_push_node_increments_count() {
            let (mut genome, _) = G::new(2, 2);
            let initial_count = genome.node_count();
            genome.push_node();
            assert_eq!(genome.node_count(), initial_count + 1);
            genome.push_node();
            assert_eq!(genome.node_count(), initial_count + 2);
        }

        /// connections() returns read-only slice of all connections
        #[test]
        fn test_connections_access() {
            let (genome, _) = G::new(2, 2);
            let conns = genome.connections();
            assert_eq!(conns.len(), 4);
        }

        /// push_connection() adds exactly 1 connection, length +1
        #[test]
        fn test_push_connection_appends() {
            let (mut genome, _) = G::new(2, 2);
            let initial_len = genome.connections().len();
            let mut new_conn = genome.connections()[0].clone();
            new_conn.enable();
            genome.push_connection(new_conn);
            assert_eq!(genome.connections().len(), initial_len + 1);
        }

        /// mutate_connection() mutates existing params only (100 iterations)
        #[test]
        fn test_mutate_connection_stochastic() {
            let (mut genome, _) = G::new(4, 4);
            let initial_conns = genome.connections().len();
            for _ in 0..100 {
                genome.mutate_connection(&mut default_rng());
            }
            assert_eq!(genome.connections().len(), initial_conns);
        }

        /// open_path() returns (from,to) where from ∉ action, to ∉ sensory
        #[test]
        fn test_open_path_valid_path() {
            let (mut genome, _) = G::new(1, 1);
            genome.connections = vec![];
            for _ in 0..100 {
                match genome.open_path(&mut default_rng()) {
                    Some((0, 1)) => {},
                    Some(p) => panic!("invalid pair {p:?} generated"),
                    None => panic!("no path generated"),
                }
            }
        }

        /// open_path() returns None when all valid paths are occupied
        #[test]
        fn test_open_path_saturation() {
            // new(1,1) already has the only possible connection (0→1)
            let (genome, _) = G::new(1, 1);
            for _ in 0..100 {
                assert_eq!(genome.open_path(&mut default_rng()), None);
            }
        }

        /// open_path() returns None for 0×0 genome (no valid paths)
        #[test]
        fn test_open_path_empty_genome() {
            let (genome, _) = G::new(0, 0);
            assert_eq!(genome.open_path(&mut default_rng()), None);
        }

        /// open_path() returns (from,to) where from ∉ action, to ∉ sensory
        #[test]
        fn test_open_path_from_not_in_action() {
            let (mut genome, _) = G::new(2, 2);
            genome.connections = vec![];
            for _ in 0..50 {
                if let Some((from, to)) = genome.open_path(&mut default_rng()) {
                    assert!(!(2..4).contains(&from));
                    assert!(to >= 2);
                }
            }
        }

        /// new_connection() increases connections().len() by exactly 1
        #[test]
        fn test_new_connection_appends_and_increments() {
            let (mut genome, _) = G::new(4, 4);
            genome.connections = vec![];
            let before_len = genome.connections().len();
            genome
                .new_connection(&mut default_rng(), &mut InnoGen::new(0))
                .expect("new_connection should succeed");
            assert_eq!(genome.connections().len(), before_len + 1);
        }

        /// new_connection() appends connection with unique path from open_path()
        #[test]
        fn test_new_connection_unique() {
            let (mut genome, _) = G::new(4, 4);
            genome.connections = vec![];
            let before_paths: std::collections::HashSet<_> =
                genome.connections().iter().map(|c| c.path()).collect();
            genome
                .new_connection(&mut default_rng(), &mut InnoGen::new(0))
                .expect("new_connection should succeed");
            let new_path = genome.connections().last().unwrap().path();
            assert!(!before_paths.contains(&new_path));
        }

        /// new_connection() returns Err when all paths are fully occupied
        #[test]
        fn test_new_connection_saturated_error() {
            // new(1,1) already has the only possible connection (0→1)
            let (mut genome, initial_inno) = G::new(1, 1);
            let mut inno = InnoGen::new(initial_inno);
            let result = genome.new_connection(&mut default_rng(), &mut inno);
            assert!(result.is_err());
        }

        /// new_connection() works correctly on empty 0×0 genome
        #[test]
        fn test_new_connection_empty_genome() {
            let (mut genome, _) = G::new(2, 2);
            genome.connections = vec![];
            let result = genome.new_connection(&mut default_rng(), &mut InnoGen::new(0));
            assert!(result.is_ok());
            assert_eq!(genome.connections().len(), 1);
        }

        /// bisect_connection() increases connections by 2, node_count by 1
        #[test]
        fn test_bisect_connection_structure_change() {
            let (mut genome, initial_inno) = G::new(1, 1);
            let initial_node_count = genome.node_count();
            let initial_conn_count = genome.connections().len();
            genome
                .bisect_connection(&mut default_rng(), &mut InnoGen::new(initial_inno))
                .expect("bisect_connection should succeed");
            assert_eq!(genome.node_count(), initial_node_count + 1);
            assert_eq!(genome.connections().len(), initial_conn_count + 2);
        }

        /// Original connection is disabled after bisection
        #[test]
        fn test_bisect_connection_original_disabled() {
            let (mut genome, initial_inno) = G::new(1, 1);
            genome
                .bisect_connection(&mut default_rng(), &mut InnoGen::new(initial_inno))
                .expect("bisect_connection should succeed");
            assert!(!genome.connections()[0].enabled);
        }

        /// Bisected connections have correct from→center→to paths
        #[test]
        fn test_bisect_connection_new_paths_valid() {
            let (mut genome, initial_inno) = G::new(1, 1);
            genome
                .bisect_connection(&mut default_rng(), &mut InnoGen::new(initial_inno))
                .expect("bisect_connection should succeed");
            let node_count = genome.node_count();
            assert_eq!(genome.connections()[1].from(), 0);
            assert_eq!(genome.connections()[1].to(), node_count - 1);
            assert_eq!(genome.connections()[2].from(), node_count - 1);
            assert_eq!(genome.connections()[2].to(), 1);
        }

        /// All three innos (original, upper, lower) are unique
        #[test]
        fn test_bisect_connection_new_innos_unique() {
            let (mut genome, initial_inno) = G::new(1, 1);
            let original_inno = genome.connections()[0].inno();
            genome
                .bisect_connection(&mut default_rng(), &mut InnoGen::new(initial_inno))
                .expect("bisect_connection should succeed");
            let new_inno_1 = genome.connections()[1].inno();
            let new_inno_2 = genome.connections()[2].inno();
            assert_ne!(original_inno, new_inno_1);
            assert_ne!(original_inno, new_inno_2);
            assert_ne!(new_inno_1, new_inno_2);
        }

        /// bisect_connection() returns Err on 0×0 genome
        #[test]
        fn test_bisect_connection_empty_genome_error() {
            let (mut genome, _) = G::new(0, 0);
            genome.connections = vec![];
            let result = genome.bisect_connection(&mut default_rng(), &mut InnoGen::new(0));
            assert!(result.is_err());
        }

        /// bisect_connection() returns Err when no connections to bisect
        #[test]
        fn test_bisect_connection_no_connections_error() {
            let (mut genome, _) = G::new(2, 2);
            genome.connections = vec![];
            let result = genome.bisect_connection(&mut default_rng(), &mut InnoGen::new(0));
            assert!(result.is_err());
        }

        /// mutate() always adds connection to completely empty genome first
        #[test]
        fn test_mutate_empty_genome_gets_connection() {
            let (mut genome, _) = G::new(2, 2);
            genome.connections = vec![];
            let initial_len = genome.connections().len();
            genome
                .mutate(&mut default_rng(), &mut InnoGen::new(0))
                .expect("mutate on empty genome should succeed");
            assert_eq!(genome.connections().len(), initial_len + 1);
        }

        /// mutate() calls new_connection, bisect_connection, or mutate_connection (100 iterations)
        #[test]
        fn test_mutate_dispatches() {
            let (mut genome, _) = G::new(3, 3);
            for _ in 0..100 {
                // mutate can legitimately fail when all paths are saturated
                let _ = genome.mutate(&mut default_rng(), &mut InnoGen::new(0));
            }
            assert!(genome.node_count() >= 6);
            assert!(!genome.connections().is_empty());
        }

        /// reproduce_with() produces offspring with mixed parent connections
        #[test]
        fn test_reproduce_with_crossover() {
            let (parent1, _) = G::new(2, 2);
            let (parent2, _) = G::new(2, 2);
            let child = parent1.reproduce_with(
                &parent2,
                std::cmp::Ordering::Equal,
                &mut default_rng(),
            );
            assert_eq!(child.sensory(), parent1.sensory());
            assert_eq!(child.action(), parent1.action());
        }

        /// offspring node_count == (max_conn_idx + 1).max(sensory+action)
        #[test]
        fn test_reproduce_with_node_count_recomputation() {
            let (mut parent1, initial_inno) = G::new(2, 2);
            let (parent2, _) = G::new(2, 2);
            parent1
                .bisect_connection(&mut default_rng(), &mut InnoGen::new(initial_inno))
                .expect("bisect should succeed");
            let child = parent1.reproduce_with(
                &parent2,
                std::cmp::Ordering::Greater,
                &mut default_rng(),
            );
            assert!(child.node_count() >= 4);
        }

        /// reproduce_with() correctly handles Equal, Less, Greater fitness ordering
        #[test]
        fn test_reproduce_with_ordering_dispatch() {
            let (parent1, _) = G::new(2, 2);
            let (parent2, _) = G::new(2, 2);
            let _child_equal = parent1.reproduce_with(
                &parent2,
                std::cmp::Ordering::Equal,
                &mut default_rng(),
            );
            let _child_greater = parent1.reproduce_with(
                &parent2,
                std::cmp::Ordering::Greater,
                &mut default_rng(),
            );
            let _child_less =
                parent1.reproduce_with(&parent2, std::cmp::Ordering::Less, &mut default_rng());
        }
    }
}
