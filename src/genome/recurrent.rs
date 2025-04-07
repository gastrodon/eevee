use super::{Connection, Genome, Node, NodeKind};
use crate::{
    crossover::crossover,
    serialize::{deserialize_connections, deserialize_nodes},
};
use core::cmp::{max, Ordering};
use rand::{seq::IteratorRandom, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A genome that allows recurrent connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recurrent<N: Node, C: Connection<N>> {
    sensory: usize,
    action: usize,
    #[serde(deserialize_with = "deserialize_nodes")]
    nodes: Vec<N>,
    #[serde(deserialize_with = "deserialize_connections")]
    connections: Vec<C>,
}

impl<N: Node, C: Connection<N>> Genome<N, C> for Recurrent<N, C> {
    fn new(sensory: usize, action: usize) -> (Self, usize) {
        let mut nodes = Vec::with_capacity(sensory + action + 1);
        for _ in 0..sensory {
            nodes.push(N::new(NodeKind::Sensory));
        }
        for _ in sensory..sensory + action {
            nodes.push(N::new(NodeKind::Action));
        }
        nodes.push(N::new(NodeKind::Static));

        (
            Self {
                sensory,
                action,
                nodes,
                connections: vec![],
            },
            (sensory + 1) * action,
        )
    }

    fn sensory(&self) -> std::ops::Range<usize> {
        0..self.sensory
    }

    fn action(&self) -> std::ops::Range<usize> {
        self.sensory..self.sensory + self.action
    }

    fn nodes(&self) -> &[N] {
        &self.nodes
    }

    fn nodes_mut(&mut self) -> &mut [N] {
        &mut self.nodes
    }

    fn push_node(&mut self, node: N) {
        self.nodes.push(node);
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
            let from = (0..self.nodes.len())
                .filter(|from| !saturated.contains(from))
                .choose(rng)?;

            let exclude = self
                .connections
                .iter()
                .filter_map(|c| (c.from() == from).then_some(c.to()))
                .collect::<HashSet<_>>();

            if let Some(to) = (0..self.nodes.len())
                .filter(|to| !exclude.contains(to))
                .choose(rng)
            {
                break Some((from, to));
            }

            saturated.insert(from);
        }
    }

    fn reproduce_with(&self, other: &Self, self_fit: Ordering, rng: &mut impl RngCore) -> Self {
        let connections = crossover(&self.connections, &other.connections, self_fit, rng);
        let nodes_size = connections
            .iter()
            .fold(0, |prev, c| max(prev, max(c.from(), c.to())));

        let mut nodes = Vec::with_capacity(self.sensory + self.action + 1);
        for _ in 0..self.sensory {
            nodes.push(N::new(NodeKind::Sensory));
        }
        for _ in self.sensory..self.sensory + self.action {
            nodes.push(N::new(NodeKind::Action));
        }
        nodes.push(N::new(NodeKind::Static));
        for _ in self.sensory + self.action..nodes_size {
            nodes.push(N::new(NodeKind::Internal));
        }

        debug_assert!(
            connections
                .iter()
                .fold(0, |acc, c| max(acc, max(c.from(), c.to())))
                < nodes.len()
        );

        Self {
            sensory: self.sensory,
            action: self.action,
            nodes,
            connections,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        genome::{node::BTNode, WConnection},
        random::default_rng,
        specie::InnoGen,
        test_t,
    };

    type N = BTNode;
    type C = WConnection<N>;
    type RecurrentContinuous = Recurrent<N, C>;

    test_t!(
    test_genome_creation[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(3, 2);
        assert_eq!(inno_head, 8);
        assert_eq!(genome.sensory().len(), 3);
        assert_eq!(genome.action().len(), 2);
        assert_eq!(genome.nodes().len(), 6);
        assert!(matches!(genome.nodes[0].kind(), NodeKind::Sensory));
        assert!(matches!(genome.nodes[3].kind(), NodeKind::Action));
        assert!(matches!(genome.nodes[5].kind(), NodeKind::Static));
    });

    test_t!(
    test_genome_creation_empty[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(0, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome.sensory().len(), 0);
        assert_eq!(genome.action().len(), 0);
        assert_eq!(genome.nodes().len(), 1);
        assert!(matches!(genome.nodes()[0].kind(), NodeKind::Static));
    });

    test_t!(
    test_genome_creation_only_sensory[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(3, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome.sensory().len(), 3);
        assert_eq!(genome.action().len(), 0);
        assert_eq!(genome.nodes().len(), 4);
        assert!(matches!(genome.nodes()[0].kind(), NodeKind::Sensory));
        assert!(matches!(genome.nodes()[2].kind(), NodeKind::Sensory));
        assert!(matches!(genome.nodes()[3].kind(), NodeKind::Static));
    });

    test_t!(
    test_genome_creation_only_action[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(0, 3);
        assert_eq!(inno_head, 3);
        assert_eq!(genome.sensory().len(), 0);
        assert_eq!(genome.action().len(), 3);
        assert_eq!(genome.nodes().len(), 4);
        assert!(matches!(genome.nodes()[0].kind(), NodeKind::Action));
        assert!(matches!(genome.nodes()[2].kind(), NodeKind::Action));
        assert!(matches!(genome.nodes()[3].kind(), NodeKind::Static));
    });

    test_t!( // TODO fixme: we are allowing connectinons to sensory + from bias
    test_gen_connection[T: RecurrentContinuous]() {
        let mut inno = InnoGen::new(0);
        let genome = T {
            sensory: 1,
            action: 1,
            nodes: vec![N::new(NodeKind::Sensory), N::new(NodeKind::Action)],
            connections: vec![
                C::new(0, 0, &mut inno),
                C::new(1, 1, &mut inno),
            ],
        };
        for _ in 0..100 {
            match genome.open_path(&mut default_rng()) {
                Some((0, o)) | Some((o, 0)) => assert_eq!(o, 1),
                Some(p) => unreachable!("invalid pair {p:?} gen'd"),
                None => unreachable!("no path gen'd"),
            }
        }
    });

    test_t!( // TODO fixme: we are allowing connections to sensory
    test_gen_connection_no_dupe[T: RecurrentContinuous]() {
        let mut inno = InnoGen::new(0);
        let genome = T {
            sensory: 1,
            action: 1,
            nodes: vec![N::new(NodeKind::Sensory), N::new(NodeKind::Action)],
            connections: vec![
                C::new(0, 0, &mut inno),
                C::new(0, 1, &mut inno),
                C::new(1, 1, &mut inno),
            ],
        };
        for _ in 0..100 {
            assert_eq!(genome.open_path(&mut default_rng()), Some((1, 0)));
        }
    });

    test_t!( // TODO fixme: we are allowing connections to sensory
    test_gen_connection_none_possible[T: RecurrentContinuous]() {
        let mut inno = InnoGen::new(0);
        assert_eq!(
            T {
                sensory: 0,
                action: 0,
                nodes: vec![],
                connections: vec![C::new(0, 1, &mut inno)],
            }
            .open_path(&mut default_rng()),
            None
        );
    });

    test_t!(
    test_mutate_connection[T: RecurrentContinuous]() {
        let (mut genome, _) = T::new(4, 4);
        let mut inno = InnoGen::new(0);
        genome.push_connection(C::new(0, 1, &mut inno));
        genome.push_connection(C::new(1, 2, &mut inno));

        let before = genome.clone();
        genome.new_connection(&mut default_rng(), &mut inno);

        assert_eq!(genome.connections().len(), before.connections().len() + 1);

        let tail = genome.connections().last().unwrap();
        assert!(!before.connections().iter().any(|c| c.inno() == tail.inno()));
        assert!(!before.connections().iter().any(|c| c.path() == tail.path()));
        assert_eq!(tail.weight(), 1.);
    });

    test_t!(
    test_mutate_bisection[T: RecurrentContinuous]() {
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = T::new(0, 1);

        genome.push_connection({
            let mut c = C::new(0, 1, &mut inno);
            c.mutate_param(&mut default_rng());
            c
        });

        let innogen = &mut InnoGen::new(1);
        genome.bisect_connection(&mut default_rng(), innogen);

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
    #[should_panic(expected = "no connections available to bisect")]
    test_mutate_bisection_empty_genome[T: RecurrentContinuous]() {
        let (mut genome, _) = T::new(0, 0);
        genome.bisect_connection(&mut default_rng(), &mut InnoGen::new(0));
    });

    test_t!(
    #[should_panic(expected = "no connections available to bisect")]
    test_mutate_bisection_no_connections[T: RecurrentContinuous]() {
        let (mut genome, _) = T::new(2, 2);
        genome.connections = vec![];
        genome.bisect_connection(&mut default_rng(), &mut InnoGen::new(0));
    });
}
