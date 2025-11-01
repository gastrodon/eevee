use super::{Connection, Genome, NodeKind};
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
pub struct Recurrent<C: Connection> {
    sensory: usize,
    action: usize,
    #[serde(deserialize_with = "deserialize_nodes")]
    nodes: Vec<NodeKind>,
    #[serde(deserialize_with = "deserialize_connections")]
    connections: Vec<C>,
}

impl<C: Connection> Genome<C> for Recurrent<C> {
    fn new(sensory: usize, action: usize) -> (Self, usize) {
        let mut nodes = Vec::with_capacity(sensory + action + 1);
        for _ in 0..sensory {
            nodes.push(NodeKind::Sensory);
        }
        for _ in sensory..sensory + action {
            nodes.push(NodeKind::Action);
        }
        nodes.push(NodeKind::Static);

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

    #[inline]
    fn sensory(&self) -> std::ops::Range<usize> {
        0..self.sensory
    }

    #[inline]
    fn action(&self) -> std::ops::Range<usize> {
        self.sensory..self.sensory + self.action
    }

    #[inline]
    fn nodes(&self) -> &[NodeKind] {
        &self.nodes
    }

    #[inline]
    fn nodes_mut(&mut self) -> &mut [NodeKind] {
        &mut self.nodes
    }

    #[inline]
    fn push_node(&mut self, node: NodeKind) {
        self.nodes.push(node);
    }

    #[inline]
    fn connections(&self) -> &[C] {
        &self.connections
    }

    #[inline]
    fn connections_mut(&mut self) -> &mut [C] {
        &mut self.connections
    }

    #[inline]
    fn push_connection(&mut self, connection: C) {
        self.connections.push(connection);
    }

    fn open_path(&self, rng: &mut impl RngCore) -> Option<(usize, usize)> {
        let mut saturated = HashSet::with_capacity(self.nodes.len());
        loop {
            let (from, _) = self
                .nodes()
                .iter()
                .enumerate()
                .filter(|(from, node)| {
                    !matches!(node, NodeKind::Action) && !saturated.contains(from)
                })
                .choose(rng)?;

            // Pre-allocate with estimated size based on connections
            let mut exclude = HashSet::with_capacity(self.nodes.len());
            for c in &self.connections {
                if c.from() == from {
                    exclude.insert(c.to());
                }
            }

            if let Some((to, _)) = self
                .nodes()
                .iter()
                .enumerate()
                .filter(|(to, node)| {
                    !matches!(node, NodeKind::Static | NodeKind::Sensory) && !exclude.contains(to)
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
        
        // Find max node index in one pass
        let mut nodes_size = 0;
        for c in &connections {
            nodes_size = max(nodes_size, max(c.from(), c.to()));
        }

        // Pre-allocate exact capacity needed
        let total_nodes = max(nodes_size + 1, self.sensory + self.action + 1);
        let mut nodes = Vec::with_capacity(total_nodes);
        
        // Build nodes vector efficiently
        for _ in 0..self.sensory {
            nodes.push(NodeKind::Sensory);
        }
        for _ in self.sensory..self.sensory + self.action {
            nodes.push(NodeKind::Action);
        }
        nodes.push(NodeKind::Static);
        for _ in self.sensory + self.action + 1..total_nodes {
            nodes.push(NodeKind::Internal);
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
    use crate::{genome::InnoGen, genome::WConnection, random::default_rng, test_t};

    type C = WConnection;
    type RecurrentContinuous = Recurrent<C>;

    test_t!(
    test_genome_creation[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(3, 2);
        assert_eq!(inno_head, 8);
        assert_eq!(genome.sensory().len(), 3);
        assert_eq!(genome.action().len(), 2);
        assert_eq!(genome.nodes().len(), 6);
        assert!(matches!(genome.nodes[0], NodeKind::Sensory));
        assert!(matches!(genome.nodes[3], NodeKind::Action));
        assert!(matches!(genome.nodes[5], NodeKind::Static));
    });

    test_t!(
    test_genome_creation_empty[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(0, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome.sensory().len(), 0);
        assert_eq!(genome.action().len(), 0);
        assert_eq!(genome.nodes().len(), 1);
        assert!(matches!(genome.nodes()[0], NodeKind::Static));
    });

    test_t!(
    test_genome_creation_only_sensory[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(3, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome.sensory().len(), 3);
        assert_eq!(genome.action().len(), 0);
        assert_eq!(genome.nodes().len(), 4);
        assert!(matches!(genome.nodes()[0], NodeKind::Sensory));
        assert!(matches!(genome.nodes()[2], NodeKind::Sensory));
        assert!(matches!(genome.nodes()[3], NodeKind::Static));
    });

    test_t!(
    test_genome_creation_only_action[T: RecurrentContinuous]() {
        let (genome, inno_head) = T::new(0, 3);
        assert_eq!(inno_head, 3);
        assert_eq!(genome.sensory().len(), 0);
        assert_eq!(genome.action().len(), 3);
        assert_eq!(genome.nodes().len(), 4);
        assert!(matches!(genome.nodes()[0], NodeKind::Action));
        assert!(matches!(genome.nodes()[2], NodeKind::Action));
        assert!(matches!(genome.nodes()[3], NodeKind::Static));
    });

    test_t!(
    test_gen_connection[T: RecurrentContinuous]() {
        let (mut genome, _ ) = T::new(1, 1);

        for _ in 0..100 {
            match genome.open_path(&mut default_rng()) {
                Some((0, 1)) | Some((2, 1)) => {}, // sensory -> action, bias -> action
                Some(p) => unreachable!("invalid pair {p:?} gen'd"),
                None => unreachable!("no path gen'd"),
            }
        }

        genome.push_connection(C::new(2, 1, &mut InnoGen::new(0)));
        for _ in 0..100 {
            assert_eq!(genome.open_path(&mut default_rng()), Some((0, 1)));
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
