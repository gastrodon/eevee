use super::{Discrete, FromGenome, Network};
use crate::{Connection, Forward, Genome};

/// A binary feedforward network. Weights are binarized at construction: `sign(w)`.
/// Internal neurons activate via `sign(sum)`; action neurons output the raw sum,
/// giving real-valued output for fitness evaluation.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
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

        topo.reverse(); // TODO can we avoid this?
        topo.into_iter()
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
