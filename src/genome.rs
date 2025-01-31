use rand::{rngs::ThreadRng, seq::IteratorRandom};
use std::{
    collections::{HashMap, HashSet},
    iter,
    sync::{Arc, Mutex},
};

fn inno_gen() -> impl Fn((usize, usize)) -> usize {
    let head = Arc::new(Mutex::new(0));
    let inno = Arc::new(Mutex::new(HashMap::<(usize, usize), usize>::new()));
    return move |v: (usize, usize)| {
        let mut head = head.lock().unwrap();
        let mut inno = inno.lock().unwrap();
        match inno.get(&v) {
            Some(n) => *n,
            None => {
                let n = *head;
                *head += 1;
                inno.insert(v, n);
                n
            }
        }
    };
}

#[derive(Debug, Clone)]
pub enum Node {
    Sensory,
    Action,
    Bias(f64),
    Internal,
}

#[derive(Debug)]
pub struct Connection {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Debug)]
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
        let inno = inno_gen();
        assert_eq!(inno((0, 1)), 0);
        assert_eq!(inno((1, 2)), 1);
        assert_eq!(inno((0, 1)), 0);
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
}
