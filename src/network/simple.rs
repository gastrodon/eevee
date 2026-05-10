use super::{FromGenome, Network};
use crate::{Connection, Genome};
use core::ops::Range;

/// A simple neural network, because man, what the fuck is going on. lol
/// Walks through connections oldest to newest, evaluating them on a flat state
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", serde(bound = "C: Connection"))]
pub struct Simple<C: Connection> {
    pub(crate) connections: Vec<C>,
    pub(crate) bias: Vec<f64>,
    #[cfg_attr(feature = "serialize", serde(skip_serializing))]
    pub(crate) state: Vec<f64>,
    #[cfg_attr(feature = "serialize", serde(skip_serializing))]
    pub(crate) sensory: Range<usize>,
    #[cfg_attr(feature = "serialize", serde(skip_serializing))]
    pub(crate) action: Range<usize>,
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
            bias: vec![0.; genome.node_count()],
            state: vec![0.; genome.node_count()],
            sensory: genome.sensory(),
            action: genome.action(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::{self, connection::BWConnection, WConnection};
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,
        G: genome::Recurrent<C>,

        /// all connections copied from genome (including disabled)
        #[test]
        fn test_preserves_all_connections() {
            let (genome, _) = G::new(2, 2);
            let nn = Simple::from_genome(&genome);

            assert_eq!(nn.connections.len(), genome.connections().len());
        }

        /// enabled/disabled status preserved from genome
        #[test]
        fn test_preserves_enabled_status() {
            let (mut genome, _) = G::new(2, 2);
            if let Some(c) = genome.connections_mut().first_mut() {
                c.disable();
            }

            let nn = Simple::from_genome(&genome);
            assert!(!nn.connections[0].enabled());
            assert!(nn.connections[1].enabled());
        }

        /// state/bias vectors match node count
        #[test]
        fn test_state_size() {
            let (genome, _) = G::new(3, 2);
            let nn = Simple::from_genome(&genome);

            assert_eq!(nn.state.len(), genome.node_count());
            assert_eq!(nn.bias.len(), genome.node_count());
        }

        /// input range matches sensory neurons
        #[test]
        fn test_input_range() {
            let (genome, _) = G::new(3, 2);
            let nn = Simple::from_genome(&genome);

            assert_eq!(nn.sensory.start, genome.sensory().start);
            assert_eq!(nn.sensory.end, genome.sensory().end);
            assert_eq!(nn.sensory.len(), genome.sensory().len());
        }

        /// state initialized to zeros
        #[test]
        fn test_initial_state() {
            let (genome, _) = G::new(2, 2);
            let nn = Simple::from_genome(&genome);

            assert!(nn.state.iter().all(|&x| x == 0.0));
        }

        /// flush() clears state vector
        #[test]
        fn test_flush_clears_state() {
            let (genome, _) = G::new(2, 2);
            let mut nn = Simple::from_genome(&genome);
            let input = vec![1.0, 0.5];

            nn.step(2, &input, |x| x);
            let state_before = nn.state.clone();

            nn.flush();
            let state_after = nn.state.clone();

            assert!(state_before.iter().any(|&x| x != 0.0));
            assert!(state_after.iter().all(|&x| x == 0.0));
        }
    }
}
