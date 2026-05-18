use super::{Discrete, FromGenome, Network};
use crate::{Connection, Forward, Genome};

/// A feedforward network built from any [Forward] genome via topological sort.
///
/// Activation is applied **per neuron** (not per connection), matching
/// the standard NEAT forward-pass formulation. A single topological pass
/// is exact by definition — there are no integration steps.
///
/// Node layout (inherited from the genome convention):
///   `[0..sensory)` sensory, `[sensory..sensory+action)` action,
///   `sensory+action` static bias, `(sensory+action+1..)` internal.
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// Neuron evaluations in topological order.
    /// Each entry is `(node_idx, [(from_idx, weight), ...])`.
    /// Sensory and static nodes are excluded — they are seeded directly in `step`.
    eval_order: Vec<(usize, Vec<(usize, f64)>)>,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
    /// Flat activation state indexed by original genome node index.
    state: Vec<f64>,
}

impl FeedForward {
    /// Build `eval_order` from a guaranteed-acyclic genome using iterative DFS post-order.
    fn build_eval_order(
        n: usize,
        sensory_end: usize,
        edges: &[(usize, usize, f64)],
    ) -> Vec<(usize, Vec<(usize, f64)>)> {
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        let mut incoming: Vec<Vec<(usize, f64)>> = vec![vec![]; n];

        for &(from, to, w) in edges {
            adj[from].push(to);
            incoming[to].push((from, w));
        }

        let mut visited = vec![false; n];
        let mut topo: Vec<usize> = Vec::with_capacity(n);
        let mut stack: Vec<(usize, bool)> = Vec::new();

        for start in 0..n {
            if !visited[start] {
                stack.push((start, false));
                while let Some((u, done)) = stack.pop() {
                    if done {
                        topo.push(u);
                        continue;
                    }
                    if visited[u] {
                        continue;
                    }
                    visited[u] = true;
                    stack.push((u, true));
                    for &v in &adj[u] {
                        if !visited[v] {
                            stack.push((v, false));
                        }
                    }
                }
            }
        }
        topo.reverse();

        topo.into_iter()
            .filter(|&i| i >= sensory_end)
            .map(|i| (i, incoming[i].clone()))
            .collect()
    }
}

impl Network for FeedForward {
    fn step<F: Fn(f64) -> f64>(&mut self, input: &[f64], σ: F) {
        self.state.fill(0.);
        self.state[self.sensory.0..self.sensory.1].copy_from_slice(input);

        for (node, incoming) in &self.eval_order {
            let sum: f64 = incoming.iter().map(|&(from, w)| self.state[from] * w).sum();
            self.state[*node] = σ(sum);
        }
    }

    fn output(&self) -> &[f64] {
        &self.state[self.action.0..self.action.1]
    }
}

impl crate::Forward for FeedForward {}
impl Discrete for FeedForward {}

impl<C: Connection, G: Genome<C> + Forward> FromGenome<C, G> for FeedForward {
    fn from_genome(genome: &G) -> Self {
        let n = genome.node_count();
        let sensory_end = genome.sensory().end;
        let edges: Vec<(usize, usize, f64)> = genome
            .connections()
            .iter()
            .filter(|c| c.enabled())
            .map(|c| (c.from(), c.to(), c.weight()))
            .collect();
        let eval_order = Self::build_eval_order(n, sensory_end, &edges);
        Self {
            eval_order,
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
        genome::{connection::BWConnection, Genome, NonRecurrent, WConnection},
        network::ToNetwork,
    };
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,
        G: NonRecurrent<C>,

        /// basic feedforward network construction
        #[test]
        fn test_feedforward_simple() {
            let (genome, _) = G::new(2, 1);
            let mut nn: FeedForward = genome.network();
            nn.step(&[1.0, 0.5], activate::steep_sigmoid);
            assert_eq!(nn.output().len(), 1);
        }

        /// feedforward network is stateless
        #[test]
        fn test_feedforward_stateless() {
            let (genome, _) = G::new(2, 1);
            let mut nn: FeedForward = genome.network();
            nn.step(&[1.0, 0.5], activate::steep_sigmoid);
            let first: Vec<f64> = nn.output().to_vec();
            nn.step(&[1.0, 0.5], activate::steep_sigmoid);
            assert_eq!(nn.output(), first.as_slice());
        }

        /// bias node affects output
        #[test]
        fn test_bias_node_contributes() {
            let (genome, _) = G::new(0, 1);
            let mut nn: FeedForward = genome.network();
            nn.step(&[], activate::steep_sigmoid);
            if !genome.connections().is_empty() {
                assert!(nn.output()[0] != 0.0);
            }
        }

        /// behavior consistency: observed output from a known network with hidden neurons
        #[test]
        fn test_feedforward_behavior_consistent() {
            // Direct edge construction with hidden neurons (6, 7).
            // Nodes: 0-2 sensory, 3-5 action, 6-7 hidden
            let edges: Vec<(usize, usize, f64)> = vec![
                (0, 6, 0.5),  // sensory[0] → hidden[0]
                (6, 3, 0.5),  // hidden[0] → action[0]
                (6, 4, 0.5),  // hidden[0] → action[1]
                (1, 3, 0.5),  // sensory[1] → action[0]
                (1, 7, 0.5),  // sensory[1] → hidden[1]
                (7, 5, 0.5),  // hidden[1] → action[2]
                (2, 4, 0.5),  // sensory[2] → action[1]
                (2, 5, 0.5),  // sensory[2] → action[2]
            ];

            let n = 8; // total nodes
            let sensory_end = 3;

            let mut nn = FeedForward {
                eval_order: FeedForward::build_eval_order(n, sensory_end, &edges),
                sensory: (0, 3),
                action: (3, 6),
                state: vec![0.0; n],
            };

            // Test inputs with expected outputs
            let cases: [([f64; 3], Vec<f64>); 3] = [
                ([1.0, 0.5, -0.5], vec![0.9701242069008241, 0.7369886984280752, 0.6612139184123204]),
                ([-1.0, 0.5, 1.0], vec![0.8052795345597723, 0.9336788971186889, 0.987178260913925]),
                ([0.0, 1.0, -1.0], vec![0.9752773002196243, 0.22705774060326145, 0.45149689483528455]),
            ];

            for (i, (input, expected)) in cases.iter().enumerate() {
                nn.step(input, activate::steep_sigmoid);
                let output = nn.output();
                assert_eq!(output.len(), 3);
                assert_eq!(output, expected.as_slice(), "case {}: input: {:?}, got: {:?}", i, input, output);
            }
        }
    }
}
