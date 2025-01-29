#[derive(Debug, Clone)]
pub enum Node {
    Sensory,
    Action,
    Bias(usize),
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
        nodes.push(Node::Bias(1));

        Self {
            sensory,
            action,
            nodes,
            connections: Vec::new(),
        }
    }
}
#[cfg(test)]
mod test {
    use super::*;

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
}
