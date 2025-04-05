/// A genome describing a Continuous Time Recurrent Neural Network (CTRNN)
use super::{Connection, Genome, Node, NodeKind};
use crate::{
    crossover::crossover,
    serialize::{deserialize_connections, deserialize_nodes},
    Happens,
};
use core::cmp::{max, Ordering};
use rand::{seq::IteratorRandom, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTRGenome<N: Node, C: Connection<N>> {
    sensory: usize,
    action: usize,
    #[serde(deserialize_with = "deserialize_nodes")]
    nodes: Vec<N>,
    #[serde(deserialize_with = "deserialize_connections")]
    connections: Vec<C>,
}

impl<N: Node, C: Connection<N>> Genome<N, C> for CTRGenome<N, C> {
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

    fn mutate_params(&mut self, rng: &mut (impl RngCore + Happens)) {
        for conn in self.connections.iter_mut() {
            conn.mutate_params(rng);
        }
    }

    fn open_path(&self, rng: &mut (impl RngCore + Happens)) -> Option<(usize, usize)> {
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

    fn reproduce_with(
        &self,
        other: &Self,
        self_fit: Ordering,
        rng: &mut (impl RngCore + Happens),
    ) -> Self {
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
        assert_f64_approx,
        genome::{node::NonBNode, WConnection},
        network::{Continuous, ToNetwork},
        random::{default_rng, percent, EvolutionEvent, ProbBinding, ProbStatic},
        specie::InnoGen,
    };
    use rulinalg::matrix::BaseMatrix;

    #[test]
    fn test_genome_creation() {
        let (genome, inno_head) = CTRGenome::<NonBNode, WConnection<NonBNode>>::new(3, 2);
        assert_eq!(inno_head, 8);
        assert_eq!(genome.sensory, 3);
        assert_eq!(genome.action, 2);
        assert_eq!(genome.nodes.len(), 6);
        assert!(matches!(genome.nodes[0], NonBNode::Sensory));
        assert!(matches!(genome.nodes[3], NonBNode::Action));
        assert!(matches!(genome.nodes[5], NonBNode::Static(_)));

        let (genome_empty, inno_head) = CTRGenome::<NonBNode, WConnection<NonBNode>>::new(0, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome_empty.sensory, 0);
        assert_eq!(genome_empty.action, 0);
        assert_eq!(genome_empty.nodes.len(), 1);
        assert!(matches!(genome_empty.nodes[0], NonBNode::Static(_)));

        let (genome_only_sensory, inno_head) =
            CTRGenome::<NonBNode, WConnection<NonBNode>>::new(3, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome_only_sensory.sensory, 3);
        assert_eq!(genome_only_sensory.action, 0);
        assert_eq!(genome_only_sensory.nodes.len(), 4);
        assert!(matches!(genome_only_sensory.nodes[0], NonBNode::Sensory));
        assert!(matches!(genome_only_sensory.nodes[2], NonBNode::Sensory));
        assert!(matches!(genome_only_sensory.nodes[3], NonBNode::Static(_)));

        let (genome_only_action, inno_head) =
            CTRGenome::<NonBNode, WConnection<NonBNode>>::new(0, 3);
        assert_eq!(inno_head, 3);
        assert_eq!(genome_only_action.sensory, 0);
        assert_eq!(genome_only_action.action, 3);
        assert_eq!(genome_only_action.nodes.len(), 4);
        assert!(matches!(genome_only_action.nodes[0], NonBNode::Action));
        assert!(matches!(genome_only_action.nodes[2], NonBNode::Action));
        assert!(matches!(genome_only_action.nodes[3], NonBNode::Static(_)));
    }

    #[test]
    fn test_gen_connection() {
        let mut inno = InnoGen::new(0);
        let genome = CTRGenome {
            sensory: 1,
            action: 1,
            nodes: vec![NonBNode::Sensory, NonBNode::Action],
            connections: vec![
                WConnection::<NonBNode>::new(0, 0, &mut inno),
                WConnection::<NonBNode>::new(1, 1, &mut inno),
            ],
        };
        for _ in 0..100 {
            match genome.open_path(&mut ProbBinding::new(ProbStatic::default(), default_rng())) {
                Some((0, o)) | Some((o, 0)) => assert_eq!(o, 1),
                Some(p) => unreachable!("invalid pair {p:?} gen'd"),
                None => unreachable!("no path gen'd"),
            }
        }
    }

    #[test]
    fn test_gen_connection_no_dupe() {
        let mut inno = InnoGen::new(0);
        let genome = CTRGenome {
            sensory: 1,
            action: 1,
            nodes: vec![NonBNode::Sensory, NonBNode::Action],
            connections: vec![
                WConnection::<NonBNode>::new(0, 0, &mut inno),
                WConnection::<NonBNode>::new(0, 1, &mut inno),
                WConnection::<NonBNode>::new(1, 1, &mut inno),
            ],
        };
        for _ in 0..100 {
            assert_eq!(
                genome.open_path(&mut ProbBinding::new(ProbStatic::default(), default_rng()),),
                Some((1, 0))
            );
        }
    }

    #[test]
    fn test_gen_connection_none_possble() {
        let mut inno = InnoGen::new(0);
        assert_eq!(
            CTRGenome::<NonBNode, WConnection<NonBNode>> {
                sensory: 0,
                action: 0,
                nodes: vec![],
                connections: vec![WConnection::<NonBNode>::new(0, 1, &mut inno)],
            }
            .open_path(&mut ProbBinding::new(ProbStatic::default(), default_rng()),),
            None
        );
    }

    #[test]
    fn test_gen_connection_saturated() {
        assert_eq!(
            CTRGenome {
                sensory: 2,
                action: 2,
                nodes: vec![
                    NonBNode::Action,
                    NonBNode::Action,
                    NonBNode::Sensory,
                    NonBNode::Sensory,
                    NonBNode::Static(1.),
                ],
                connections: (0..5)
                    .flat_map(|from| {
                        (0..5).map(move |to| {
                            WConnection::<NonBNode>::new(from, to, &mut InnoGen::new(0))
                        })
                    })
                    .collect(),
            }
            .open_path(&mut ProbBinding::new(ProbStatic::default(), default_rng()),),
            None
        )
    }

    #[test]
    fn test_mutate_connection() {
        let (mut genome, _) = CTRGenome::<NonBNode, WConnection<NonBNode>>::new(4, 4);
        let mut inno = InnoGen::new(0);
        genome.push_2_connections(
            WConnection::<NonBNode>::new(0, 1, &mut inno),
            WConnection::<NonBNode>::new(1, 2, &mut inno),
        );

        let before = genome.clone();
        genome.mutate_connection(
            &mut ProbBinding::new(ProbStatic::default(), default_rng()),
            &mut inno,
        );

        assert_eq!(genome.connections().len(), before.connections().len() + 1);

        let tail = genome.connections().last().unwrap();
        assert!(!before.connections().iter().any(|c| c.inno() == tail.inno()));
        assert!(!before.connections().iter().any(|c| c.path() == tail.path()));
        assert_eq!(tail.weight(), 1.);
    }

    #[test]
    fn test_mutate_bisection() {
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = CTRGenome::<NonBNode, WConnection<NonBNode>>::new(0, 1);

        genome.push_connection({
            let mut c = WConnection::<NonBNode>::new(0, 1, &mut inno);
            c.mutate_params(&mut ProbBinding::new(
                ProbStatic::default().with_overrides(&[(EvolutionEvent::NewWeight, percent(100))]),
                default_rng(),
            ));
            c
        });

        let innogen = &mut InnoGen::new(1);
        genome.mutate_bisection(
            &mut ProbBinding::new(ProbStatic::default(), default_rng()),
            innogen,
        );

        let connections: &[WConnection<NonBNode>] = genome.connections();
        assert!(!connections[0].enabled);

        assert_eq!(connections[1].from(), 0);
        assert_eq!(connections[1].to(), 2);
        assert_eq!(connections[1].weight(), 1.0);
        assert!(connections[1].enabled);
        assert_eq!(
            genome.connections[1].inno,
            innogen.path((genome.connections[1].from(), genome.connections[1].to()))
        );

        assert_eq!(genome.connections[2].from(), 2);
        assert_eq!(genome.connections[2].to(), 1);
        assert_eq!(genome.connections[1].weight(), 1.);
        assert_eq!(
            genome.connections[2].weight(),
            genome.connections[0].weight()
        );
        assert!(genome.connections[2].enabled);
        assert_eq!(
            genome.connections[2].inno,
            innogen.path((genome.connections[2].from(), genome.connections[2].to()))
        );

        assert_ne!(genome.connections[0].inno, genome.connections[1].inno);
        assert_ne!(genome.connections[1].inno, genome.connections[2].inno);
        assert_ne!(genome.connections[0].inno, genome.connections[2].inno);
    }

    #[test]
    #[should_panic]
    fn test_mutate_bisection_empty_genome() {
        let (mut genome, _) = CTRGenome::<NonBNode, WConnection<NonBNode>>::new(0, 0);
        genome.mutate_bisection(
            &mut ProbBinding::new(ProbStatic::default(), default_rng()),
            &mut InnoGen::new(0),
        );
    }

    #[test]
    #[should_panic]
    fn test_mutate_bisection_no_connections() {
        let (mut genome, _) = CTRGenome::<NonBNode, WConnection<NonBNode>>::new(2, 2);
        genome.connections = vec![];
        genome.mutate_bisection(
            &mut ProbBinding::new(ProbStatic::default(), default_rng()),
            &mut InnoGen::new(0),
        );
    }

    #[test]
    fn test_state_head() {
        let mut state = vec![0.; 5];
        {
            let size = 3;
            let state: &mut [f64] = &mut state;
            assert!(state.len() >= size);
            &mut state[0..size]
        }
        .clone_from_slice(&[1., 2., 3.]);
        assert_eq!(state, vec![1., 2., 3., 0., 0.])
    }

    #[test]
    fn test_ctrgenome_network() {
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = CTRGenome::<NonBNode, WConnection<NonBNode>>::new(2, 2);
        genome.connections = vec![
            WConnection::<NonBNode>::new(0, 3, &mut inno),
            WConnection::<NonBNode>::new(0, 1, &mut inno),
            WConnection::<NonBNode>::new(0, 1, &mut inno),
        ];

        let nn: Continuous = genome.network();
        unsafe {
            for WConnection::<NonBNode> {
                from, to, weight, ..
            } in genome.connections.iter().filter(|c| c.enabled)
            {
                assert_f64_approx!(nn.w.get_unchecked([*from, *to]), weight);
            }

            for (i, node) in genome.nodes.iter().enumerate() {
                assert_f64_approx!(
                    nn.θ.get_unchecked([0, i]),
                    if let NonBNode::Static(b) = node {
                        b
                    } else {
                        &0.
                    }
                )
            }
        }

        for i in nn.sensory.0..nn.sensory.1 {
            assert!(genome
                .nodes
                .get(i)
                .is_some_and(|n| matches!(n, NonBNode::Sensory)))
        }
        for i in nn.action.0..nn.action.1 {
            assert!(genome
                .nodes
                .get(i)
                .is_some_and(|n| matches!(n, NonBNode::Action)))
        }
    }
}
