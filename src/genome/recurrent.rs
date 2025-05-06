use super::{Connection, ConnectionPoint, Genome, NodeKind};
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

macro_rules! unbound_nodes {
    ($self:ident, $saturated:ident, $matcher:pat) => {
        $self
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(node_idx, node)| {
                (matches!(node, $matcher)
                    && !($saturated).is_some_and(|exclude| exclude.contains(&node_idx)))
                .then_some(node_idx)
            })
    };
    (@from: $self:ident, $saturated:ident) => {
        unbound_nodes!(
            $self,
            $saturated,
            NodeKind::Sensory | NodeKind::Internal | NodeKind::Static
        )
    };
    (@to: $self:ident, $saturated:ident) => {
        unbound_nodes!($self, $saturated, NodeKind::Internal | NodeKind::Action)
    };
}

macro_rules! bound_nodes {
    ($self:ident, $filter_map:expr, $matcher:pat) => {{
        let exclude = $self
            .connections
            .iter()
            .filter_map($filter_map)
            .collect::<HashSet<_>>();

        $self
            .nodes()
            .iter()
            .enumerate()
            .filter_map(move |(node_idx, node)| {
                (matches!(node, $matcher) && !exclude.contains(&node_idx)).then_some(node_idx)
            })
    }};
    (@from: $self:ident, $bound:expr) => {
        bound_nodes!(
            $self,
            |c| (c.to() == $bound).then_some(c.from()),
            NodeKind::Sensory | NodeKind::Internal | NodeKind::Static
        )
    };
    (@to: $self:ident, $bound:expr) => {
        bound_nodes!(
            $self,
            |c| (c.from() == $bound).then_some(c.to()),
            NodeKind::Internal | NodeKind::Action
        )
    };
}

impl<C: Connection> Recurrent<C> {
    fn unbound_from(
        &self,
        saturated: Option<&HashSet<usize>>,
        rng: &mut impl RngCore,
    ) -> Option<usize> {
        unbound_nodes!(@from: self, saturated).choose(rng)
    }

    #[allow(dead_code, reason = "may be useful for generating paths in reverse")]
    fn unbound_to(
        &self,
        saturated: Option<&HashSet<usize>>,
        rng: &mut impl RngCore,
    ) -> Option<usize> {
        unbound_nodes!(@to: self, saturated).choose(rng)
    }

    fn bound_from(&self, to_idx: usize, rng: &mut impl RngCore) -> Option<usize> {
        debug_assert!(
            matches!(
                self.nodes.get(to_idx),
                Some(NodeKind::Internal | NodeKind::Action)
            ),
            "node[{to_idx}] {:?} cannot be bound as to",
            self.nodes.get(to_idx)
        );
        bound_nodes!(@from: self, to_idx).choose(rng)
    }

    fn bound_to(&self, from_idx: usize, rng: &mut impl RngCore) -> Option<usize> {
        debug_assert!(
            matches!(
                self.nodes.get(from_idx),
                Some(NodeKind::Sensory | NodeKind::Internal | NodeKind::Static)
            ),
            "node[{from_idx}] {:?} cannot be bound as from",
            self.nodes.get(from_idx),
        );
        bound_nodes!(@to: self, from_idx).choose(rng)
    }
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

    fn sensory(&self) -> std::ops::Range<usize> {
        0..self.sensory
    }

    fn action(&self) -> std::ops::Range<usize> {
        self.sensory..self.sensory + self.action
    }

    fn nodes(&self) -> &[NodeKind] {
        &self.nodes
    }

    fn nodes_mut(&mut self) -> &mut [NodeKind] {
        &mut self.nodes
    }

    fn push_node(&mut self, node: NodeKind) {
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

    fn open_path<'a>(
        &self,
        included: Option<ConnectionPoint>,
        rng: &mut impl RngCore,
    ) -> Option<(usize, usize)> {
        match included {
            Some(ConnectionPoint::From(from_idx)) => self
                .bound_to(from_idx, rng)
                .map(|to_idx| (from_idx, to_idx)),
            Some(ConnectionPoint::To(to_idx)) => self
                .bound_from(to_idx, rng)
                .map(|from_idx| (from_idx, to_idx)),
            None => {
                let mut saturated = HashSet::new();
                loop {
                    let from_idx = self.unbound_from(Some(&saturated), rng)?;
                    if let Some(to_idx) = self.bound_to(from_idx, rng) {
                        break Some((from_idx, to_idx));
                    } else {
                        saturated.insert(from_idx);
                    }
                }
            }
        }
    }

    fn reproduce_with(&self, other: &Self, self_fit: Ordering, rng: &mut impl RngCore) -> Self {
        let connections = crossover(&self.connections, &other.connections, self_fit, rng);
        let nodes_size = connections
            .iter()
            .fold(0, |prev, c| max(prev, max(c.from(), c.to())));

        let mut nodes = Vec::with_capacity(self.sensory + self.action + 1);
        for _ in 0..self.sensory {
            nodes.push(NodeKind::Sensory);
        }
        for _ in self.sensory..self.sensory + self.action {
            nodes.push(NodeKind::Action);
        }
        nodes.push(NodeKind::Static);
        for _ in self.sensory + self.action..nodes_size {
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
            match genome.open_path(None, &mut default_rng()) {
                Some((0, 1)) | Some((2, 1)) => {}, // sensory -> action, bias -> action
                Some(p) => unreachable!("invalid pair {p:?} ({:?} -> {:?}) gen'd", genome.nodes()[p.0], genome.nodes()[p.1]),
                None => unreachable!("no path gen'd"),
            }
        }

        genome.push_connection(C::new(2, 1, &mut InnoGen::new(0)));
        for _ in 0..100 {
            assert_eq!(genome.open_path(None, &mut default_rng()), Some((0, 1)));
        }
    });

    test_t!(
    test_gen_connection_including[T: RecurrentContinuous]() {
        let (mut genome, _) = T::new(1, 1);

        for _ in 0..100 {
            assert_eq!(genome.open_path(Some(ConnectionPoint::From(0)), &mut default_rng()), Some((0, 1)));
            match genome.open_path(Some(ConnectionPoint::To(1)), &mut default_rng()) {
                Some((0, 1)) | Some((2, 1)) => {}, // sensory -> action, bias -> action
                Some(p) => unreachable!("invalid pair {p:?} ({:?} -> {:?}) gen'd", genome.nodes()[p.0], genome.nodes()[p.1]),
                None => unreachable!("no path gen'd"),
            }
        }

        genome.push_connection(C::new(2, 1, &mut InnoGen::new(0)));
        for _ in 0..100 {
            assert_eq!(genome.open_path(Some(ConnectionPoint::To(1)), &mut default_rng()), Some((0, 1)));
        }

    });

    test_t!(
    #[should_panic(expected = "node[0] Some(Sensory) cannot be bound as to")]
    test_gen_connection_invalid_to[T: RecurrentContinuous]() {
        T::new(1, 1).0.open_path(Some(ConnectionPoint::To(0)), &mut default_rng());
    });

    test_t!(
    #[should_panic(expected = "node[1] Some(Action) cannot be bound as from")]
    test_gen_connection_invalid_from[T: RecurrentContinuous]() {
        T::new(1, 1).0.open_path(Some(ConnectionPoint::From(1)), &mut default_rng());
    });

    test_t!(
    test_gen_connection_none_possible[T: RecurrentContinuous]() {
        let (genome, _) = T::new(0, 0);
        assert_eq!(
            genome
            .open_path(None, &mut default_rng()),
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

        // source connection
        let source = &genome.connections()[0];
        assert!(!source.enabled);

        // new pair front half: from -> center
        let front = &genome.connections()[1];
        assert_eq!(front.from(), 0);
        assert_eq!(front.to(), 2);
        assert_eq!(front.weight(), 1.);
        assert!(front.enabled);
        assert_eq!(
            front.inno,
            innogen.path((front.from(), front.to()))
        );

        // new pair rear half: center -> from
        let rear = &genome.connections()[2];
        assert_eq!(rear.from(), 2);
        assert_eq!(rear.to(), 1);
        assert_eq!(
            rear.weight(),
            source.weight()
        );
        assert!(rear.enabled);
        assert_eq!(
            rear.inno,
            innogen.path((rear.from(), rear.to()))
        );


        // external  connection: * -> center
        let inter = &genome.connections()[3];
        assert_eq!(HashSet::from([source.inno, front.inno, rear.inno, inter.inno]).len(), 4)
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
