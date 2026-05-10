use super::{Connection, Genome, InnoGen};
use crate::{crossover::crossover, serialize::deserialize_connections};
use core::cmp::{max, Ordering};
use rand::{seq::IteratorRandom, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A genome that allows recurrent connections
///
/// Node layout: `[0..sensory)` sensory, `[sensory..sensory+action)` action,
/// `(sensory+action..)` internal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recurrent<C: Connection> {
    sensory: usize,
    action: usize,
    node_count: usize,
    #[serde(deserialize_with = "deserialize_connections")]
    connections: Vec<C>,
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
    use crate::{genome::WConnection, random::default_rng, test_t};

    type C = WConnection;
    type RecurrentContinuous = Recurrent<C>;

    test_t!(
    test_genome_creation[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(3, 2);
        assert_eq!(inno_head, 6);
        assert_eq!(genome.sensory().len(), 3);
        assert_eq!(genome.action().len(), 2);
        assert_eq!(genome.node_count(), 5);
    });

    test_t!(
    test_genome_creation_empty[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(0, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome.sensory().len(), 0);
        assert_eq!(genome.action().len(), 0);
        assert_eq!(genome.node_count(), 0);
    });

    test_t!(
    test_genome_creation_only_sensory[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(3, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome.sensory().len(), 3);
        assert_eq!(genome.action().len(), 0);
        assert_eq!(genome.node_count(), 3);
    });

    test_t!(
    test_genome_creation_only_action[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(0, 3);
        assert_eq!(inno_head, 0);
        assert_eq!(genome.sensory().len(), 0);
        assert_eq!(genome.action().len(), 3);
        assert_eq!(genome.node_count(), 3);
    });

    test_t!(
    test_gen_connection[T: RecurrentContinuous]() {
        let (mut genome, _ ) = T::new(1, 1);
        genome.connections = vec![]; // TODO generalize empty connection state

        for _ in 0..100 {
            match genome.open_path(&mut default_rng()) {
                Some((0, 1)) => {}, // sensory -> action
                Some(p) => unreachable!("invalid pair {p:?} gen'd"),
                None => unreachable!("no path gen'd"),
            }
        }

        genome.push_connection(C::new(0, 1, &mut InnoGen::new(0)));
        for _ in 0..100 {
            assert_eq!(genome.open_path(&mut default_rng()), None);
        }
    });

    test_t!(
    test_gen_connection_none_possible[T: RecurrentContinuous]() {
        let (genome, _) = T::new(0, 0);
        assert_eq!(
            genome
            .open_path(&mut default_rng()),
            None
        );
    });

    test_t!(
    test_mutate_connection[T: RecurrentContinuous]() {
        let (mut genome, _) = T::new(4, 4);
        let mut inno = InnoGen::new(0);
        genome.connections = vec![]; // TODO generalize empty connection state
        genome.push_connection(C::new(0, 1, &mut inno));
        genome.push_connection(C::new(1, 2, &mut inno));

        let before = genome.clone();
        genome.new_connection(&mut default_rng(), &mut inno).unwrap_or_else(|e| panic!("failed new_connection: {e}"));

        assert_eq!(genome.connections().len(), before.connections().len() + 1);

        let tail = genome.connections().last().unwrap();
        assert!(!before.connections().iter().any(|c| c.inno() == tail.inno()));
        assert!(!before.connections().iter().any(|c| c.path() == tail.path()));
        assert_eq!(tail.weight(), 1.);
    });

    test_t!(
    test_mutate_bisection[T: RecurrentContinuous]() {
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = T::new(1, 1);

        genome.connections = vec![]; // TODO generalize empty connection state
        genome.push_connection({
            let mut c = C::new(0, 1, &mut inno);
            c.mutate_param(&mut default_rng());
            c
        });

        let innogen = &mut InnoGen::new(1);
        genome.bisect_connection(&mut default_rng(), innogen).unwrap_or_else(|e| panic!("failed new_connection: {e}"));

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
    });

    test_t!(
    test_mutate_bisection_empty_genome[T: RecurrentContinuous]() {
        let (mut genome, _) = T::new(0, 0);
        genome.connections = vec![]; // TODO generalize empty connection state
        assert!(genome.bisect_connection(&mut default_rng(), &mut InnoGen::new(0)).is_err());
    });

    test_t!(
    test_mutate_bisection_no_connections[T: RecurrentContinuous]() {
        let (mut genome, _) = T::new(2, 2);
        genome.connections = vec![]; // TODO generalize empty connection state
        assert!(genome.bisect_connection(&mut default_rng(), &mut InnoGen::new(0)).is_err());
    });
}
