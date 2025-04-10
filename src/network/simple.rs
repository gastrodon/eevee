use super::{FromGenome, Network};
use crate::{genome::NodeKind, serialize::deserialize_connections, Connection, Genome};
use core::ops::Range;
use serde::{Deserialize, Serialize};

/// A simple neural network, because man, what the fuck is going on. lol
/// Walks through connections oldest to newest, evaluating them on a flat state
#[derive(Debug, Serialize, Deserialize)]
pub struct Simple<C: Connection> {
    #[serde(deserialize_with = "deserialize_connections")]
    connections: Vec<C>, // TODO this is copying because of deserialization
    bias: Vec<f64>,
    #[serde(skip_serializing)]
    state: Vec<f64>,
    #[serde(skip_serializing)]
    sensory: Range<usize>,
    #[serde(skip_serializing)]
    action: Range<usize>,
}

impl<C: Connection> Network for Simple<C> {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        debug_assert!(input.len() == self.sensory.len());
        self.state[self.sensory.start..self.sensory.end].copy_from_slice(input);
        if !self.connections.is_empty() {
            for _ in 0..prec {
                for c in self.connections.iter() {
                    self.state[c.to()] +=
                        σ((self.bias[c.from()] + self.state[c.from()]) * c.weight())
                }
            }
        }
    }

    fn flush(&mut self) {
        self.state = vec![0.; self.state.len()];
    }

    fn output(&self) -> &[f64] {
        &self.state[self.action.start..self.action.end]
    }
}

impl<C: Connection, G: Genome<C>> FromGenome<C, G> for Simple<C> {
    fn from_genome(genome: &G) -> Self {
        Simple {
            connections: genome.connections().to_owned(),
            bias: genome
                .nodes()
                .iter()
                .map(|n| {
                    if matches!(n, NodeKind::Static) {
                        1.
                    } else {
                        0.
                    }
                })
                .collect(),
            state: vec![0.; genome.nodes().len()],
            sensory: genome.sensory(),
            action: genome.action(),
        }
    }
}
