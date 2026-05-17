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
        genome::{connection::BWConnection, InnoGen, NonRecurrent, WConnection},
        network::ToNetwork,
    };
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,

        /// basic feedforward network construction
        #[test]
        fn test_feedforward_simple() {
            let (genome, _) = NonRecurrent::<C>::new(2, 1);
            let mut nn: FeedForward = genome.network();
            nn.step(&[1.0, 0.5], activate::steep_sigmoid);
            assert_eq!(nn.output().len(), 1);
        }

        /// feedforward network is stateless
        #[test]
        fn test_feedforward_stateless() {
            let (genome, _) = NonRecurrent::<C>::new(2, 1);
            let mut nn: FeedForward = genome.network();
            nn.step(&[1.0, 0.5], activate::steep_sigmoid);
            let first: Vec<f64> = nn.output().to_vec();
            nn.step(&[1.0, 0.5], activate::steep_sigmoid);
            assert_eq!(nn.output(), first.as_slice());
        }

        /// bias node affects output
        #[test]
        fn test_bias_node_contributes() {
            let (genome, _) = NonRecurrent::<C>::new(0, 1);
            let mut nn: FeedForward = genome.network();
            nn.step(&[], activate::steep_sigmoid);
            if !genome.connections().is_empty() {
                assert!(nn.output()[0] != 0.0);
            }
        }

        /// topological ordering is respected
        #[test]
        fn test_topo_order_respected() {
            let mut inno = InnoGen::new(0);
            let (mut genome, _) = NonRecurrent::<C>::new(1, 1);
            genome.push_node(); // internal node at index 2
            genome
                .connections_mut()
                .iter_mut()
                .for_each(|c| c.disable());
            genome.push_connection(C::new(0, 2, &mut inno)); // sensory→hidden
            genome.push_connection(C::new(2, 1, &mut inno)); // hidden→action

            let mut nn: FeedForward = genome.network();
            nn.step(&[1.0], activate::steep_sigmoid);
            assert!(nn.output()[0] != 0.0);
        }
    }
}
