/// A genome describing a Continuous Time Recurrent Neural Network (CTRNN)
use crate::{
    crossover::crossover,
    genome::{Connection, Genome, Node, NodeKind},
    network::FromGenome,
    specie::InnoGen,
    Ctrnn, Happens,
};
use core::{
    cmp::{max, Ordering},
    hash::Hash,
};
use rand::{seq::IteratorRandom, Rng, RngCore};
use rand_distr::StandardNormal;
use rulinalg::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CTRNode {
    Sensory,
    Action,
    Bias(f64),
    Internal,
}

impl Node for CTRNode {
    fn new(kind: NodeKind) -> Self {
        match kind {
            NodeKind::Sensory => Self::Sensory,
            NodeKind::Action => Self::Action,
            NodeKind::Internal => Self::Internal,
            NodeKind::Bias => Self::Bias(1.),
        }
    }

    fn kind(&self) -> super::NodeKind {
        match self {
            Self::Sensory => NodeKind::Sensory,
            Self::Action => NodeKind::Action,
            Self::Bias(_) => NodeKind::Bias,
            Self::Internal => NodeKind::Internal,
        }
    }

    fn bias(&self) -> f64 {
        match self {
            Self::Bias(b) => *b,
            _ => 0.,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CTRConnection {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub enabled: bool,
}

impl Connection for CTRConnection {
    const EXCESS_COEFFICIENT: f64 = 1.0;
    const DISJOINT_COEFFICIENT: f64 = 1.0;
    const PARAM_COEFFICIENT: f64 = 0.4;

    type Node = CTRNode;

    fn new(from: usize, to: usize, inno: &mut InnoGen) -> Self {
        Self {
            inno: inno.path((from, to)),
            from,
            to,
            weight: 1.,
            enabled: true,
        }
    }

    fn inno(&self) -> usize {
        self.inno
    }

    fn enable(&mut self) {
        self.enabled = true;
    }

    fn disable(&mut self) {
        self.enabled = false;
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn bisect(&mut self, center: usize, inno: &mut InnoGen) -> (Self, Self) {
        self.disable();
        (
            // from -{1.}> bisect-node
            CTRConnection {
                inno: inno.path((self.from, center)),
                from: self.from,
                to: center,
                weight: 1.,
                enabled: true,
            },
            // bisect-node -{w}> to
            CTRConnection {
                inno: inno.path((center, self.to)),
                from: center,
                to: self.to,
                weight: self.weight,
                enabled: true,
            },
        )
    }

    fn param_diff(&self, other: &Self) -> f64 {
        // TODO add other ctrnn specific diffs when we have those fields available
        // theta, bias, weight
        (self.weight - other.weight).abs()
    }
}

impl Default for CTRConnection {
    fn default() -> Self {
        Self {
            inno: 0,
            from: 0,
            to: 0,
            weight: 0.,
            enabled: true,
        }
    }
}

impl Hash for CTRConnection {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inno.hash(state);
        self.from.hash(state);
        self.to.hash(state);
        ((1000. * self.weight) as usize).hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTRGenome {
    sensory: usize,
    action: usize,
    nodes: Vec<CTRNode>,
    connections: Vec<CTRConnection>,
}

impl CTRGenome {
    const MUTATE_WEIGHT_FAC: f64 = 0.05;
}

impl Genome for CTRGenome {
    type Connection = CTRConnection;

    fn new(sensory: usize, action: usize) -> (Self, usize) {
        let mut nodes = Vec::with_capacity(sensory + action + 1);
        for _ in 0..sensory {
            nodes.push(CTRNode::Sensory);
        }
        for _ in sensory..sensory + action {
            nodes.push(CTRNode::Action);
        }
        nodes.push(CTRNode::Bias(1.));

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

    fn nodes(&self) -> &[Self::Node] {
        &self.nodes
    }

    fn push_node(&mut self, node: Self::Node) {
        self.nodes.push(node);
    }

    fn connections(&self) -> &[Self::Connection] {
        &self.connections
    }

    fn connections_mut(&mut self) -> &mut [Self::Connection] {
        &mut self.connections
    }

    fn push_connection(&mut self, connection: Self::Connection) {
        self.connections.push(connection);
    }

    fn mutate_params(&mut self, rng: &mut (impl RngCore + Happens)) {
        for conn in self.connections.iter_mut() {
            if rng.random_ratio(1, 10) {
                conn.weight = rng.sample(StandardNormal);
            } else {
                conn.weight += Self::MUTATE_WEIGHT_FAC * rng.sample::<f64, _>(StandardNormal)
            }
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
                .filter_map(|c| (c.from == from).then_some(c.to))
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
        other: &CTRGenome,
        self_fit: Ordering,
        rng: &mut (impl RngCore + Happens),
    ) -> Self {
        let connections = crossover(&self.connections, &other.connections, self_fit, rng);
        let nodes_size = connections
            .iter()
            .fold(0, |prev, CTRConnection { from, to, .. }| {
                max(prev, max(*from, *to))
            });

        let mut nodes = Vec::with_capacity(self.sensory + self.action + 1);
        for _ in 0..self.sensory {
            nodes.push(CTRNode::Sensory);
        }
        for _ in self.sensory..self.sensory + self.action {
            nodes.push(CTRNode::Action);
        }
        nodes.push(CTRNode::Bias(1.));
        for _ in self.sensory + self.action..nodes_size {
            nodes.push(CTRNode::Internal);
        }

        debug_assert!(
            connections
                .iter()
                .fold(0, |acc, c| max(acc, max(c.from, c.to)))
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

impl FromGenome<CTRGenome> for Ctrnn {
    fn from_genome(genome: &CTRGenome) -> Self {
        let cols = genome.nodes.len();
        Ctrnn {
            y: Matrix::zeros(1, cols),
            θ: Matrix::new(
                1,
                cols,
                genome
                    .nodes
                    .iter()
                    .map(|n| if let CTRNode::Bias(b) = n { *b } else { 0. })
                    .collect::<Vec<_>>(),
            ),
            τ: Matrix::ones(1, cols),
            w: {
                let mut w = vec![0.; cols * cols];
                for CTRConnection {
                    from, to, weight, ..
                } in genome.connections.iter().filter(|c| c.enabled)
                {
                    w[from * cols + to] = *weight;
                }
                Matrix::new(cols, cols, w)
            },
            sensory: (0, genome.sensory),
            action: (genome.sensory, genome.sensory + genome.action),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        assert_f64_approx,
        network::ToNetwork,
        random::{default_rng, ProbBinding, ProbStatic},
        specie::InnoGen,
    };
    use rulinalg::matrix::BaseMatrix;

    #[test]
    fn test_genome_creation() {
        let (genome, inno_head) = CTRGenome::new(3, 2);
        assert_eq!(inno_head, 8);
        assert_eq!(genome.sensory, 3);
        assert_eq!(genome.action, 2);
        assert_eq!(genome.nodes.len(), 6);
        assert!(matches!(genome.nodes[0], CTRNode::Sensory));
        assert!(matches!(genome.nodes[3], CTRNode::Action));
        assert!(matches!(genome.nodes[5], CTRNode::Bias(_)));

        let (genome_empty, inno_head) = CTRGenome::new(0, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome_empty.sensory, 0);
        assert_eq!(genome_empty.action, 0);
        assert_eq!(genome_empty.nodes.len(), 1);
        assert!(matches!(genome_empty.nodes[0], CTRNode::Bias(_)));

        let (genome_only_sensory, inno_head) = CTRGenome::new(3, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome_only_sensory.sensory, 3);
        assert_eq!(genome_only_sensory.action, 0);
        assert_eq!(genome_only_sensory.nodes.len(), 4);
        assert!(matches!(genome_only_sensory.nodes[0], CTRNode::Sensory));
        assert!(matches!(genome_only_sensory.nodes[2], CTRNode::Sensory));
        assert!(matches!(genome_only_sensory.nodes[3], CTRNode::Bias(_)));

        let (genome_only_action, inno_head) = CTRGenome::new(0, 3);
        assert_eq!(inno_head, 3);
        assert_eq!(genome_only_action.sensory, 0);
        assert_eq!(genome_only_action.action, 3);
        assert_eq!(genome_only_action.nodes.len(), 4);
        assert!(matches!(genome_only_action.nodes[0], CTRNode::Action));
        assert!(matches!(genome_only_action.nodes[2], CTRNode::Action));
        assert!(matches!(genome_only_action.nodes[3], CTRNode::Bias(_)));
    }

    #[test]
    fn test_gen_connection() {
        let genome = CTRGenome {
            sensory: 1,
            action: 1,
            nodes: vec![CTRNode::Sensory, CTRNode::Action],
            connections: vec![
                CTRConnection {
                    inno: 0,
                    from: 0,
                    to: 0,
                    weight: 0.,
                    enabled: true,
                },
                CTRConnection {
                    inno: 1,
                    from: 1,
                    to: 1,
                    weight: 0.,
                    enabled: true,
                },
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
        let genome = CTRGenome {
            sensory: 1,
            action: 1,
            nodes: vec![CTRNode::Sensory, CTRNode::Action],
            connections: vec![
                CTRConnection {
                    inno: 0,
                    from: 0,
                    to: 0,
                    weight: 1.,
                    enabled: true,
                },
                CTRConnection {
                    inno: 1,
                    from: 0,
                    to: 1,
                    weight: 1.,
                    enabled: true,
                },
                CTRConnection {
                    inno: 2,
                    from: 1,
                    to: 1,
                    weight: 1.,
                    enabled: true,
                },
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
        assert_eq!(
            CTRGenome {
                sensory: 0,
                action: 0,
                nodes: vec![],
                connections: vec![CTRConnection {
                    inno: 0,
                    from: 0,
                    to: 1,
                    weight: 1.,
                    enabled: true,
                }],
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
                    CTRNode::Action,
                    CTRNode::Action,
                    CTRNode::Sensory,
                    CTRNode::Sensory,
                    CTRNode::Bias(1.),
                ],
                connections: (0..5)
                    .flat_map(|from| {
                        (0..5).map(move |to| CTRConnection {
                            inno: 0,
                            from,
                            to,
                            weight: 1.,
                            enabled: true,
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
        let (mut genome, _) = CTRGenome::new(4, 4);
        let mut inext = InnoGen::new(0);
        genome.connections = vec![
            CTRConnection {
                inno: inext.path((0, 1)),
                from: 0,
                to: 1,
                weight: 0.5,
                enabled: true,
            },
            CTRConnection {
                inno: inext.path((1, 2)),
                from: 1,
                to: 2,
                weight: 0.5,
                enabled: true,
            },
        ];

        let before = genome.clone();
        genome.mutate_connection(
            &mut ProbBinding::new(ProbStatic::default(), default_rng()),
            &mut inext,
        );

        assert_eq!(genome.connections.len(), before.connections.len() + 1);

        let tail = genome.connections.last().unwrap();
        assert!(!before.connections.iter().any(|c| c.inno == tail.inno));
        assert!(!before
            .connections
            .iter()
            .any(|c| (c.from, c.to) == (tail.from, tail.to)));
        assert_eq!(tail.weight, 1.);
    }

    #[test]
    fn test_mutate_bisection() {
        let (mut genome, _) = CTRGenome::new(0, 1);
        genome.connections = vec![CTRConnection {
            inno: 0,
            from: 0,
            to: 1,
            weight: 0.5,
            enabled: true,
        }];
        let innogen = &mut InnoGen::new(1);
        genome.mutate_bisection(
            &mut ProbBinding::new(ProbStatic::default(), default_rng()),
            innogen,
        );

        assert!(!genome.connections[0].enabled);

        assert_eq!(genome.connections[1].from, 0);
        assert_eq!(genome.connections[1].to, 2);
        assert_eq!(genome.connections[1].weight, 1.0);
        assert!(genome.connections[1].enabled);
        assert_eq!(
            genome.connections[1].inno,
            innogen.path((genome.connections[1].from, genome.connections[1].to))
        );

        assert_eq!(genome.connections[2].from, 2);
        assert_eq!(genome.connections[2].to, 1);
        assert_eq!(genome.connections[2].weight, 0.5);
        assert!(genome.connections[2].enabled);
        assert_eq!(
            genome.connections[2].inno,
            innogen.path((genome.connections[2].from, genome.connections[2].to))
        );

        assert_ne!(genome.connections[0].inno, genome.connections[1].inno);
        assert_ne!(genome.connections[1].inno, genome.connections[2].inno);
        assert_ne!(genome.connections[0].inno, genome.connections[2].inno);
    }

    #[test]
    #[should_panic]
    fn test_mutate_bisection_empty_genome() {
        let (mut genome, _) = CTRGenome::new(0, 0);
        genome.mutate_bisection(
            &mut ProbBinding::new(ProbStatic::default(), default_rng()),
            &mut InnoGen::new(0),
        );
    }

    #[test]
    #[should_panic]
    fn test_mutate_bisection_no_connections() {
        let (mut genome, _) = CTRGenome::new(2, 2);
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
        let (mut genome, _) = CTRGenome::new(2, 2);
        genome.connections = vec![
            CTRConnection {
                inno: 0,
                from: 0,
                to: 3,
                weight: 0.5,
                enabled: true,
            },
            CTRConnection {
                inno: 1,
                from: 0,
                to: 1,
                weight: -1.,
                enabled: true,
            },
            CTRConnection {
                inno: 2,
                from: 0,
                to: 1,
                weight: 1.2,
                enabled: false,
            },
        ];

        let nn: Ctrnn = genome.network();
        unsafe {
            for CTRConnection {
                from, to, weight, ..
            } in genome.connections.iter().filter(|c| c.enabled)
            {
                assert_f64_approx!(nn.w.get_unchecked([*from, *to]), weight);
            }

            for (i, node) in genome.nodes.iter().enumerate() {
                assert_f64_approx!(
                    nn.θ.get_unchecked([0, i]),
                    if let CTRNode::Bias(b) = node { b } else { &0. }
                )
            }
        }

        for i in nn.sensory.0..nn.sensory.1 {
            assert!(genome
                .nodes
                .get(i)
                .is_some_and(|n| matches!(n, CTRNode::Sensory)))
        }
        for i in nn.action.0..nn.action.1 {
            assert!(genome
                .nodes
                .get(i)
                .is_some_and(|n| matches!(n, CTRNode::Action)))
        }
    }
}
