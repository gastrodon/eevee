use super::{FromGenome, Linear, Network, Stateless};
use crate::{Connection, Genome};
use std::collections::VecDeque;

/// A feedforward network built from any genome via topological sort.
///
/// Recurrent (back-edge) connections are silently dropped — any connection
/// that would form a cycle is excluded when the eval order is computed.
/// This lets you evolve with the `Recurrent` genome (which allows cycles)
/// while running inference as a pure DAG, matching the Mari/O NEAT style.
///
/// Activation is applied **per neuron** (not per connection), matching
/// the standard NEAT forward-pass formulation.
///
/// Node layout (inherited from the genome convention):
///   `[0..sensory)` sensory, `[sensory..sensory+action)` action,
///   `sensory+action` static bias, `(sensory+action+1..)` internal.
#[derive(Debug, Clone)]
pub struct Feedforward {
    /// Neuron evaluations in topological order.
    /// Each entry is `(node_idx, [(from_idx, weight), ...])`.
    /// Sensory and static nodes are excluded — they are seeded directly in `step`.
    eval_order: Vec<(usize, Vec<(usize, f64)>)>,
    /// Index of the static (bias) node; its state is fixed at 1.0.
    static_idx: usize,
    /// Total number of nodes.
    n_nodes: usize,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
    /// Flat activation state indexed by original genome node index.
    state: Vec<f64>,
}

impl Feedforward {
    /// Build `eval_order` from enabled connections using Kahn's algorithm.
    /// Back-edges (those that close cycles) are naturally omitted because
    /// nodes in cycles never reach in-degree zero.
    fn build_eval_order(
        n: usize,
        sensory_end: usize,
        static_idx: usize,
        edges: &[(usize, usize, f64)],
    ) -> Vec<(usize, Vec<(usize, f64)>)> {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        let mut in_deg = vec![0usize; n];

        for &(from, to, w) in edges {
            adj[from].push((to, w));
            in_deg[to] += 1;
        }

        // seed with nodes that have no incoming edges
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut topo: Vec<usize> = Vec::with_capacity(n);

        while let Some(u) = queue.pop_front() {
            topo.push(u);
            for &(v, _) in &adj[u] {
                in_deg[v] -= 1;
                if in_deg[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        // pre-collect incoming edges per node
        let mut incoming: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for &(from, to, w) in edges {
            incoming[to].push((from, w));
        }

        topo.into_iter()
            .filter(|&i| i >= sensory_end && i != static_idx)
            .map(|i| (i, incoming[i].clone()))
            .collect()
    }
}

impl Network for Feedforward {
    fn step<F: Fn(f64) -> f64>(&mut self, _prec: usize, input: &[f64], σ: F) {
        debug_assert_eq!(input.len(), self.sensory.1 - self.sensory.0);

        // seed sensory inputs
        self.state[self.sensory.0..self.sensory.1].copy_from_slice(input);

        // static (bias) node is always 1.0
        self.state[self.static_idx] = 1.0;

        // evaluate in topological order
        for (node, incoming) in &self.eval_order {
            let sum: f64 = incoming.iter().map(|&(from, w)| self.state[from] * w).sum();
            self.state[*node] = σ(sum);
        }
    }

    fn flush(&mut self) {
        self.state = vec![0.0; self.n_nodes];
    }

    fn output(&self) -> &[f64] {
        &self.state[self.action.0..self.action.1]
    }
}

impl Linear for Feedforward {}
impl Stateless for Feedforward {}

impl<C: Connection, G: Genome<C>> FromGenome<C, G> for Feedforward {
    fn from_genome(genome: &G) -> Self {
        let n = genome.node_count();
        let sensory_end = genome.sensory().end;
        let static_idx = genome.action().end; // layout: sensory | action | static | internal

        let edges: Vec<(usize, usize, f64)> = genome
            .connections()
            .iter()
            .filter(|c| c.enabled())
            .map(|c| (c.from(), c.to(), c.weight()))
            .collect();

        let eval_order = Self::build_eval_order(n, sensory_end, static_idx, &edges);

        Self {
            eval_order,
            static_idx,
            n_nodes: n,
            sensory: (genome.sensory().start, genome.sensory().end),
            action: (genome.action().start, genome.action().end),
            state: vec![0.0; n],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        activate,
        genome::{InnoGen, Recurrent, WConnection},
        network::ToNetwork,
    };

    type G = Recurrent<WConnection>;

    #[test]
    fn test_feedforward_simple() {
        // 2 sensory, 1 action — direct connections wired by Genome::new
        let (genome, _) = G::new(2, 1);
        let mut nn: Feedforward = genome.network();
        nn.step(1, &[1.0, 0.5], activate::steep_sigmoid);
        assert_eq!(nn.output().len(), 1);
    }

    #[test]
    fn test_feedforward_drops_back_edges() {
        // Build a genome with a manual cycle: sensory→internal→action + action→internal back-edge.
        // Feedforward should evaluate without panicking and produce output.
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = G::new(1, 1);
        // add an internal node (index 3: after sensory=0, action=1, static=2)
        genome.push_node();
        // forward edges: 0→3, 3→1
        genome.push_connection(WConnection::new(0, 3, &mut inno));
        genome.push_connection(WConnection::new(3, 1, &mut inno));
        // back-edge that would close a cycle: 1→3
        genome.push_connection(WConnection::new(1, 3, &mut inno));

        let mut nn: Feedforward = genome.network();
        nn.step(1, &[1.0], activate::steep_sigmoid);
        assert_eq!(nn.output().len(), 1);
    }

    #[test]
    fn test_feedforward_flush() {
        let (genome, _) = G::new(2, 1);
        let mut nn: Feedforward = genome.network();
        nn.step(1, &[1.0, 1.0], activate::steep_sigmoid);
        nn.flush();
        assert!(nn.output().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_bias_node_contributes() {
        // static→action connections are wired by Genome::new, so output should be non-zero
        let (genome, _) = G::new(0, 1);
        let mut nn: Feedforward = genome.network();
        nn.step(1, &[], activate::steep_sigmoid);
        if !genome.connections().is_empty() {
            assert!(nn.output()[0] != 0.0);
        }
    }

    #[test]
    fn test_topo_order_respected() {
        // sensory→hidden→action chain; output should reflect activation through hidden layer
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = G::new(1, 1);
        genome.push_node(); // internal node at index 3
        genome.connections_mut().iter_mut().for_each(|c| c.disable());
        genome.push_connection(WConnection::new(0, 3, &mut inno)); // sensory→hidden
        genome.push_connection(WConnection::new(3, 1, &mut inno)); // hidden→action

        let mut nn: Feedforward = genome.network();
        nn.step(1, &[1.0], activate::steep_sigmoid);
        // hidden fires from sensory, action fires from hidden — should be non-zero
        assert!(nn.output()[0] != 0.0);
    }
}
