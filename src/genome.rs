use crate::eval::{steep_sigmoid, Game};
use rand::{rngs::ThreadRng, seq::IteratorRandom, Rng};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    hash::Hash,
    iter,
    sync::{Arc, Mutex},
};

pub struct InnoGen {
    pub head: usize,
    seen: HashMap<(usize, usize), usize>,
}

impl InnoGen {
    pub fn new(head: usize) -> Self {
        Self {
            head,
            seen: HashMap::new(),
        }
    }

    pub fn new_arc(head: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::new(head)))
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

#[derive(Debug, Clone)]
pub enum Node {
    Sensory,
    Action,
    Bias(f64),
    Internal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Connection {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct Genome {
    pub sensory: usize,
    pub action: usize,
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
}

impl Genome {
    pub fn new(sensory: usize, action: usize) -> Self {
        let mut nodes = Vec::with_capacity(sensory + action + 1);
        for _ in 0..sensory {
            nodes.push(Node::Sensory);
        }
        for _ in sensory..sensory + action {
            nodes.push(Node::Action);
        }
        nodes.push(Node::Bias(1.));

        Self {
            sensory,
            action,
            nodes,
            connections: Vec::new(),
        }
    }

    // picks an unconnected pair, generates a connection between them, and applies it
    // fails if no pair can be picked
    pub fn mutate_connection(
        &mut self,
        rng: &mut ThreadRng,
        inext: &mut InnoGen,
    ) -> Result<(), Box<dyn Error>> {
        if let Some((from, to)) = gen_connection(self, rng) {
            self.connections.push(Connection {
                inno: inext.path((from, to)),
                from,
                to,
                weight: 1.,
                enabled: true,
            });
            Ok(())
        } else {
            Err("connections on genome are fully saturated".into())
        }
    }

    // Picks a source connection, bisects it, and applies it
    // picked source connection is marked as disabled
    pub fn mutate_bisection(
        &mut self,
        rng: &mut ThreadRng,
        inext: &mut InnoGen,
    ) -> Result<(), Box<dyn Error>> {
        if self.connections.is_empty() {
            return Err("no connections available to bisect".into());
        }

        let pick_idx = rng.random_range(0..self.connections.len());
        let new_node_idx = self.nodes.len();
        let (lower, upper) = {
            // possibly: would it make sense for a bisection to require a new inno?
            let pick = self.connections.get_mut(pick_idx).unwrap();
            pick.enabled = false;
            (
                // from -{1.}> bisect-node
                Connection {
                    inno: inext.path((pick.from, new_node_idx)),
                    from: pick.from,
                    to: new_node_idx,
                    weight: 1.,
                    enabled: true,
                },
                // bisect-node -{w}> to
                Connection {
                    inno: inext.path((new_node_idx, pick.from)),
                    from: new_node_idx,
                    to: pick.to,
                    weight: pick.weight,
                    enabled: true,
                },
            )
        };

        self.nodes.push(Node::Internal);
        self.connections.push(lower);
        self.connections.push(upper);
        Ok(())
    }

    /// given a mutable state, propagate it with the genome's connections ye
    pub fn propagate_once(&self, state: &mut [f64]) {
        for c in self
            .connections
            .iter()
            .filter(|Connection { enabled, .. }| *enabled)
        {
            let prop = steep_sigmoid(state[c.from]) * c.weight;
            if state[c.to] == 0. {
                state[c.to] = prop
            } else {
                state[c.to] = (state[c.to] + prop) / 2. // is this even the right way to rolling avg?
            }
        }
    }

    fn propagate(self, game: impl Game) -> usize {
        todo!()
    }
}

// TODO move state_{head,tail} to a common place for game logic?
fn state_head(size: usize, state: &mut [f64]) -> &mut [f64] {
    assert!(state.len() >= size);
    &mut state[0..size]
}

fn state_tail(size: usize, state: &[f64]) -> &[f64] {
    let l = state.len();
    assert!(l >= size);
    &state[l - size..l]
}

/// Given a genome with 0 or more nodes, try to generate a connection between nodes
/// a connection should have a unique (from, to) from any other connection on genome,
/// and the connection should not describe a node that points to itself
fn gen_connection(genome: &Genome, rng: &mut ThreadRng) -> Option<(usize, usize)> {
    let mut saturated = HashSet::new();
    loop {
        let from = (0..genome.nodes.len())
            .filter(|from| !saturated.contains(from))
            .choose(rng)?;

        let exclude = genome
            .connections
            .iter()
            .filter_map(|c| (c.from == from).then_some(c.to))
            .chain(iter::once(from))
            .collect::<HashSet<_>>();

        if let Some(to) = (0..genome.nodes.len())
            .filter(|to| !exclude.contains(to))
            .choose(rng)
        {
            break Some((from, to));
        }

        saturated.insert(from);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::vec;

    #[test]
    fn test_inno_gen() {
        let mut inno = InnoGen::new(0);
        assert_eq!(inno.head, 0);
        assert_eq!(inno.path((0, 1)), 0);
        assert_eq!(inno.path((1, 2)), 1);
        assert_eq!(inno.path((0, 1)), 0);
        assert_eq!(inno.head, 2);

        let mut inno2 = InnoGen::new(inno.head);
        assert_eq!(inno2.path((1, 0)), 2);
        assert_eq!(inno2.path((0, 1)), 3);
    }

    #[test]
    fn test_genome_creation() {
        let genome = Genome::new(3, 2);
        assert_eq!(genome.sensory, 3);
        assert_eq!(genome.action, 2);
        assert_eq!(genome.nodes.len(), 6);
        assert!(matches!(genome.nodes[0], Node::Sensory));
        assert!(matches!(genome.nodes[3], Node::Action));
        assert!(matches!(genome.nodes[5], Node::Bias(_)));
        assert_eq!(genome.connections.len(), 0);

        let genome_empty = Genome::new(0, 0);
        assert_eq!(genome_empty.sensory, 0);
        assert_eq!(genome_empty.action, 0);
        assert_eq!(genome_empty.nodes.len(), 1);
        assert!(matches!(genome_empty.nodes[0], Node::Bias(_)));
        assert_eq!(genome_empty.connections.len(), 0);

        let genome_only_sensory = Genome::new(3, 0);
        assert_eq!(genome_only_sensory.sensory, 3);
        assert_eq!(genome_only_sensory.action, 0);
        assert_eq!(genome_only_sensory.nodes.len(), 4);
        assert!(matches!(genome_only_sensory.nodes[0], Node::Sensory));
        assert!(matches!(genome_only_sensory.nodes[2], Node::Sensory));
        assert!(matches!(genome_only_sensory.nodes[3], Node::Bias(_)));
        assert_eq!(genome_only_sensory.connections.len(), 0);

        let genome_only_action = Genome::new(0, 3);
        assert_eq!(genome_only_action.sensory, 0);
        assert_eq!(genome_only_action.action, 3);
        assert_eq!(genome_only_action.nodes.len(), 4);
        assert!(matches!(genome_only_action.nodes[0], Node::Action));
        assert!(matches!(genome_only_action.nodes[2], Node::Action));
        assert!(matches!(genome_only_action.nodes[3], Node::Bias(_)));
        assert_eq!(genome_only_action.connections.len(), 0);
    }

    #[test]
    fn test_gen_connection() {
        let genome = Genome {
            sensory: 1,
            action: 1,
            nodes: vec![Node::Sensory, Node::Action],
            connections: vec![],
        };
        for _ in 0..100 {
            match gen_connection(&genome, &mut rand::rng()) {
                Some((0, o)) | Some((o, 0)) => assert_eq!(o, 1),
                Some(p) => assert!(false, "invalid pair {p:?} gen'd"),
                None => assert!(false, "no path gen'd"),
            }
        }
    }

    #[test]
    fn test_gen_connection_no_dupe() {
        let genome = Genome {
            sensory: 1,
            action: 1,
            nodes: vec![Node::Sensory, Node::Action],
            connections: vec![Connection {
                inno: 0,
                from: 0,
                to: 1,
                weight: 1.,
                enabled: true,
            }],
        };
        for _ in 0..100 {
            assert_eq!(gen_connection(&genome, &mut rand::rng()), Some((1, 0)));
        }
    }

    #[test]
    fn test_gen_connection_none_possble() {
        assert_eq!(
            gen_connection(
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![Connection {
                        inno: 0,
                        from: 0,
                        to: 1,
                        weight: 1.,
                        enabled: true,
                    }],
                },
                &mut rand::rng()
            ),
            None
        );
    }

    #[test]
    fn test_gen_connection_saturated() {
        assert_eq!(
            gen_connection(
                &Genome {
                    sensory: 2,
                    action: 2,
                    nodes: vec![
                        Node::Action,
                        Node::Action,
                        Node::Sensory,
                        Node::Sensory,
                        Node::Bias(1.),
                    ],
                    connections: (0..5)
                        .flat_map(|from| {
                            (0..5).filter_map(move |to| {
                                (from != to).then_some(Connection {
                                    inno: 0,
                                    from,
                                    to,
                                    weight: 1.,
                                    enabled: true,
                                })
                            })
                        })
                        .collect(),
                },
                &mut rand::rng()
            ),
            None
        )
    }

    #[test]
    fn test_mutate_connection() {
        let mut genome = Genome::new(4, 4);
        let mut inext = InnoGen::new(0);
        genome.connections = vec![
            Connection {
                inno: inext.path((0, 1)),
                from: 0,
                to: 1,
                weight: 0.5,
                enabled: true,
            },
            Connection {
                inno: inext.path((1, 2)),
                from: 1,
                to: 2,
                weight: 0.5,
                enabled: true,
            },
        ];

        let before = genome.clone();
        genome
            .mutate_connection(&mut rand::rng(), &mut inext)
            .unwrap();

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
        let mut genome = Genome::new(0, 1);
        genome.connections = vec![Connection {
            inno: 0,
            from: 0,
            to: 1,
            weight: 0.5,
            enabled: true,
        }];
        genome
            .mutate_bisection(&mut rand::rng(), &mut InnoGen::new(0))
            .unwrap();

        assert!(!genome.connections[0].enabled);

        assert_eq!(genome.connections[1].from, 0);
        assert_eq!(genome.connections[1].to, 2);
        assert_eq!(genome.connections[1].weight, 1.0);
        assert!(genome.connections[1].enabled);

        assert_eq!(genome.connections[2].from, 2);
        assert_eq!(genome.connections[2].to, 1);
        assert_eq!(genome.connections[2].weight, 0.5);
        assert!(genome.connections[2].enabled);
    }

    #[test]
    fn test_mutate_bisection_empty_genome() {
        let mut genome = Genome::new(0, 0);
        let result = genome.mutate_bisection(&mut rand::rng(), &mut InnoGen::new(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_mutate_bisection_no_connections() {
        let mut genome = Genome::new(2, 2);
        let result = genome.mutate_bisection(&mut rand::rng(), &mut InnoGen::new(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_state_head() {
        let mut state = vec![0.; 5];
        state_head(3, &mut state).clone_from_slice(&[1., 2., 3.]);
        assert_eq!(state, vec![1., 2., 3., 0., 0.])
    }

    #[test]
    fn test_propagate_once() {
        let mut genome = Genome::new(2, 2);
        genome.connections = vec![
            Connection {
                inno: 0,
                from: 0,
                to: 3,
                weight: 0.5,
                enabled: true,
            },
            Connection {
                inno: 1,
                from: 0,
                to: 1,
                weight: -1.,
                enabled: true,
            },
        ];

        let mut state = vec![0.; genome.nodes.len()];
        state_head(genome.sensory, &mut state).clone_from_slice(&[1., 1.]);
        genome.propagate_once(&mut state);

        assert_eq!(
            state_tail(genome.action, &state),
            &[steep_sigmoid(1.) * 0.5, 0.]
        );
    }
}
