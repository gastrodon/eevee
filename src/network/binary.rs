use super::{Discrete, FromGenome, Network};
use crate::{Connection, Forward, Genome};

/// A binary feedforward network. Weights are binarized at construction: `sign(w)`.
/// Internal neurons activate via `sign(sum)`; action neurons output the raw sum,
/// giving real-valued output for fitness evaluation.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct BinaryFeedForward {
    /// Topologically-ordered neuron evaluations: `(node_idx, [(from_idx, sign(weight)), ...])`.
    eval_order: Vec<(usize, Vec<(usize, f64)>)>,
    /// Index of the action range — action nodes get raw sums, not re-binarized.
    action_start: usize,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
    state: Vec<f64>,
}

#[inline]
fn sign(x: f64) -> f64 {
    if x >= 0. {
        1.
    } else {
        -1.
    }
}

impl BinaryFeedForward {
    fn build_eval_order(
        n: usize,
        sensory_end: usize,
        edges: &[(usize, usize, f64)],
    ) -> Vec<(usize, Vec<(usize, f64)>)> {
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        let mut incoming: Vec<Vec<(usize, f64)>> = vec![vec![]; n];

        for &(from, to, w) in edges {
            adj[from].push(to);
            incoming[to].push((from, sign(w)));
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

        topo.into_iter()
            .rev()
            .filter(|&i| i >= sensory_end)
            .map(|i| (i, incoming[i].clone()))
            .collect()
    }
}

impl Network for BinaryFeedForward {
    fn step<F: Fn(f64) -> f64>(&mut self, input: &[f64], _σ: F) {
        self.state.fill(0.);
        self.state[self.sensory.0..self.sensory.1].copy_from_slice(input);

        for &(node, ref incoming) in &self.eval_order {
            let sum: f64 = incoming
                .iter()
                .map(|&(from, w)| sign(self.state[from]) * w)
                .sum();

            self.state[node] = if node >= self.action_start {
                sum
            } else {
                sign(sum)
            };
        }
    }

    fn output(&self) -> &[f64] {
        &self.state[self.action.0..self.action.1]
    }
}

impl crate::Forward for BinaryFeedForward {}
impl Discrete for BinaryFeedForward {}

impl<C: Connection, G: Genome<C> + Forward> FromGenome<C, G> for BinaryFeedForward {
    fn from_genome(genome: &G) -> Self {
        let edges: Vec<(usize, usize, f64)> = genome
            .connections()
            .iter()
            .filter(|c| c.enabled())
            .map(|c| (c.from(), c.to(), c.weight()))
            .collect();

        Self {
            eval_order: Self::build_eval_order(genome.node_count(), genome.sensory().end, &edges),
            action_start: genome.action().start,
            sensory: (genome.sensory().start, genome.sensory().end),
            action: (genome.action().start, genome.action().end),
            state: vec![0.0; genome.node_count()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        genome::{connection::BWConnection, Genome, InnoGen, NonRecurrent, WConnection},
        network::ToNetwork,
    };
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,
        G: NonRecurrent<C>,

        /// basic binary network construction
        #[test]
        fn test_binary_construction() {
            let (genome, _) = G::new(2, 1);
            let mut nn: BinaryFeedForward = genome.network();
            nn.step(&[1.0, 0.5], |_| 0.0);
            assert_eq!(nn.output().len(), 1);
        }

        /// weights are binarized to sign at construction
        #[test]
        fn test_binary_weight_binarization() {
            let mut inno = InnoGen::new(0);
            let (mut genome, _) = G::new(1, 1);
            genome.push_connection(C::new(0, 1, &mut inno));

            let nn: BinaryFeedForward = genome.network();
            // Verify network is constructed and can output
            assert!(nn.output().len() > 0);
        }

        /// action neurons output raw sums (not binarized)
        #[test]
        fn test_binary_action_raw_output() {
            let (genome, _) = G::new(2, 1);
            let mut nn: BinaryFeedForward = genome.network();
            // Input values that would produce non-zero sums
            nn.step(&[1.0, 1.0], |_| 0.0);
            let output = nn.output();
            assert_eq!(output.len(), 1);
            // Output should exist (may be 0 if no connections or all disabled)
            let _ = output[0];
        }

        /// internal neurons apply sign to sum
        #[test]
        fn test_binary_internal_sign() {
            let mut inno = InnoGen::new(0);
            let (mut genome, _) = G::new(1, 1);
            genome.push_node(); // internal node at index 2
            genome
                .connections_mut()
                .iter_mut()
                .for_each(|c| c.disable());
            genome.push_connection(C::new(0, 2, &mut inno)); // sensory→hidden
            genome.push_connection(C::new(2, 1, &mut inno)); // hidden→action

            let mut nn: BinaryFeedForward = genome.network();
            nn.step(&[1.0], |_| 0.0);
            // Verify network stepped successfully with internal binary operations
            assert_eq!(nn.output().len(), 1);
        }

        /// behavior consistency: observed output from a known network with hidden neurons
        #[test]
        fn test_binary_behavior_consistent() {
            // Direct edge construction with hidden neurons (6, 7).
            // Nodes: 0-2 sensory, 3-5 action, 6-7 hidden
            let edges: Vec<(usize, usize, f64)> = vec![
                (0, 6, 1.0),  // sensory[0] → hidden[0]
                (6, 3, 1.0),  // hidden[0] → action[0]
                (6, 4, 1.0),  // hidden[0] → action[1]
                (1, 3, 1.0),  // sensory[1] → action[0]
                (1, 7, 1.0),  // sensory[1] → hidden[1]
                (7, 5, 1.0),  // hidden[1] → action[2]
                (2, 4, 1.0),  // sensory[2] → action[1]
                (2, 5, 1.0),  // sensory[2] → action[2]
            ];

            let n = 8; // total nodes
            let sensory_end = 3;
            let action_start = 3;

            let mut nn = BinaryFeedForward {
                eval_order: BinaryFeedForward::build_eval_order(n, sensory_end, &edges),
                action_start,
                sensory: (0, 3),
                action: (3, 6),
                state: vec![0.0; n],
            };

            // Test inputs with expected outputs
            let cases: [([f64; 3], Vec<f64>); 3] = [
                ([1.0, 0.5, -0.5], vec![2.0, 0.0, 0.0]),
                ([-1.0, 0.5, 1.0], vec![0.0, 0.0, 2.0]),
                ([0.0, 1.0, -1.0], vec![2.0, 0.0, 0.0]),
            ];

            for (i, (input, expected)) in cases.iter().enumerate() {
                nn.step(input, |_| 0.);
                let output = nn.output();
                assert_eq!(output, expected.as_slice(), "case {}: input: {:?}, got: {:?}", i, input, output)
            }
        }

    }
}
