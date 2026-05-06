use super::{FromGenome, Network, Recurrent, Stateful};
use crate::{Connection, Genome};
use serde::{Deserialize, Serialize};

/// A binary neural network where activations and weights are binarized to {-1, +1}.
///
/// Weights are binarized from the genome at construction: `sign(w)`. Each step
/// binarizes activations with `sign` before propagating. The σ parameter is unused;
/// the activation function is implicitly `sign`. Action neurons receive the raw
/// integer sum (not re-binarized), giving real-valued output for fitness evaluation.
#[derive(Debug, Serialize, Deserialize)]
pub struct Binary {
    /// Pre-binarized connections: (from, to, sign(weight))
    connections: Vec<(usize, usize, f64)>,
    /// Neuron state
    y: Vec<f64>,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

#[inline]
fn sign(x: f64) -> f64 {
    if x >= 0. {
        1.
    } else {
        -1.
    }
}

impl Network for Binary {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], _σ: F) {
        debug_assert_eq!(input.len(), self.sensory.1 - self.sensory.0);
        self.y[self.sensory.0..self.sensory.1].copy_from_slice(input);
        let n = self.y.len();
        let mut next = vec![0.; n];
        for _ in 0..prec {
            next.iter_mut().for_each(|v| *v = 0.);
            for &(from, to, w_bin) in &self.connections {
                next[to] += sign(self.y[from]) * w_bin;
            }
            next[self.sensory.0..self.sensory.1].copy_from_slice(input);
            self.y.copy_from_slice(&next);
        }
    }

    fn flush(&mut self) {
        self.y.iter_mut().for_each(|v| *v = 0.);
    }

    fn output(&self) -> &[f64] {
        &self.y[self.action.0..self.action.1]
    }
}

impl Recurrent for Binary {}

impl Stateful for Binary {}

impl<C: Connection, G: Genome<C>> FromGenome<C, G> for Binary {
    fn from_genome(genome: &G) -> Self {
        Self {
            connections: genome
                .connections()
                .iter()
                .filter(|c| c.enabled())
                .map(|c| (c.from(), c.to(), sign(c.weight())))
                .collect(),
            y: vec![0.; genome.nodes().len()],
            sensory: (genome.sensory().start, genome.sensory().end),
            action: (genome.action().start, genome.action().end),
        }
    }
}
