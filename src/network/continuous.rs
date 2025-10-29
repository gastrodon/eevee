use super::{FromGenome, Recurrent, Stateful};
use crate::{
    genome::NodeKind,
    serialize::{deserialize_matrix_flat, deserialize_matrix_square, serialize_matrix},
    Connection, Genome, Network,
};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use serde::{Deserialize, Serialize};

/// A stateful NN who receives input continuously, useful for realtime problems
/// and genomes whos connections may be recurrent.
///
/// Implementation based on the network described by
/// on the dynamics of small continuous-time recurrent neural networks (beer 1995)
/// and with some code stolen from [TLmaK0's neat implentation](https://github.com/TLmaK0/rustneat)
#[derive(Debug, Serialize, Deserialize)]
pub struct Continuous {
    /// 1d state of neurons 0-N
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub y: Matrix<f64>,
    /// 1d bias of neurons 0-N
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub θ: Matrix<f64>,
    /// 1d membrane resistance time constant
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub τ: Matrix<f64>,
    /// Nd weights between neurons, indexed as [from, to]
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_square"
    )]
    pub w: Matrix<f64>,
    /// Range of input neurons, indexing into y
    pub sensory: (usize, usize),
    /// Range of output neurons, indexing into y
    pub action: (usize, usize),
}

impl Network for Continuous {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            self.y += (((&self.y + &self.θ).apply(&σ) * &self.w) - &self.y + &m_input)
                .elemul(&self.τ)
                .apply(&|v| v * inv);
        }
    }

    fn flush(&mut self) {
        self.y = Matrix::zeros(1, self.y.cols());
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}

impl Recurrent for Continuous {}

impl Stateful for Continuous {}

impl<C: Connection, G: Genome<C>> FromGenome<C, G> for Continuous {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.nodes().len();
        Self {
            y: Matrix::zeros(1, cols),
            θ: Matrix::new(
                1,
                cols,
                genome
                    .nodes()
                    .iter()
                    .map(|n| {
                        if matches!(n, NodeKind::Static) {
                            1.
                        } else {
                            0.
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            τ: Matrix::new(1, cols, vec![0.1; cols]),
            w: {
                let mut w = vec![0.; cols * cols];
                for c in genome.connections().iter().filter(|c| c.enabled()) {
                    w[c.from() * cols + c.to()] = c.weight();
                }
                Matrix::new(cols, cols, w)
            },
            sensory: (genome.sensory().start, genome.sensory().end),
            action: (genome.action().start, genome.action().end),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        activate, assert_f64_approx, assert_matrix_approx,
        genome::InnoGen,
        genome::{self, NodeKind, WConnection},
        random::default_rng,
    };
    use rand_distr::{num_traits::Float, Distribution, Uniform};
    use rulinalg::matrix::Matrix;

    // Macro for comparing f64 arrays with epsilon tolerance

    #[test]
    fn test_ctrnn_serialization_deserialization() {
        let n_neurons = 10;
        let mut rng = default_rng();
        let dist = Uniform::new(-10., 10.).unwrap();

        let mut y_data = vec![0.0; n_neurons];
        let mut theta_data = vec![0.0; n_neurons];
        let mut tau_data = vec![0.0; n_neurons];
        let mut w_data = vec![0.0; n_neurons * n_neurons];

        for i in 0..n_neurons {
            y_data[i] = dist.sample(&mut rng);
            theta_data[i] = dist.sample(&mut rng);
            tau_data[i] = dist.sample(&mut rng).abs() + 0.1;

            for j in 0..n_neurons {
                w_data[i * n_neurons + j] = dist.sample(&mut rng);
            }
        }

        let original = Continuous {
            y: Matrix::new(1, n_neurons, y_data),
            θ: Matrix::new(1, n_neurons, theta_data),
            τ: Matrix::new(1, n_neurons, tau_data),
            w: Matrix::new(n_neurons, n_neurons, w_data),
            sensory: (0, 2),
            action: (3, 5),
        };

        let serialized = original.to_string().expect("Failed to serialize");

        let deserialized = Continuous::from_str(&serialized).expect("Failed to deserialize");

        assert_matrix_approx!(original.y.data(), deserialized.y.data());
        assert_matrix_approx!(original.θ.data(), deserialized.θ.data());
        assert_matrix_approx!(original.τ.data(), deserialized.τ.data());
        assert_matrix_approx!(original.w.data(), deserialized.w.data());

        assert_eq!(original.sensory, deserialized.sensory);
        assert_eq!(original.action, deserialized.action);
    }

    #[test]
    fn test_ctrnn_behavioral_equivalence() {
        let n_neurons = 10;
        let mut rng = default_rng();
        let dist = Uniform::new(-10., 10.).unwrap();

        let mut y_data = vec![0.0; n_neurons];
        let mut θ_data = vec![0.0; n_neurons];
        let mut τ_data = vec![0.0; n_neurons];
        let mut w_data = vec![0.0; n_neurons * n_neurons];

        for i in 0..n_neurons {
            y_data[i] = dist.sample(&mut rng);
            θ_data[i] = dist.sample(&mut rng);
            τ_data[i] = dist.sample(&mut rng).abs() + 0.1;

            for j in 0..n_neurons {
                w_data[i * n_neurons + j] = dist.sample(&mut rng);
            }
        }

        let mut original = Continuous {
            y: Matrix::new(1, n_neurons, y_data),
            θ: Matrix::new(1, n_neurons, θ_data),
            τ: Matrix::new(1, n_neurons, τ_data),
            w: Matrix::new(n_neurons, n_neurons, w_data),
            sensory: (0, 2),
            action: (3, 5),
        };

        let mut deserialized =
            Continuous::from_str(&original.to_string().expect("Failed to serialize"))
                .expect("Failed to deserialize");

        let precision = 10;
        let n_steps = 500;

        for __ in 0..n_steps {
            let input: Vec<f64> = (0..2).map(|_| dist.sample(&mut rng)).collect();

            original.step(precision, &input, activate::steep_sigmoid);
            deserialized.step(precision, &input, activate::steep_sigmoid);

            let original_output = original.output();
            let deserialized_output = deserialized.output();

            assert_matrix_approx!(original_output, deserialized_output);
        }
    }

    #[test]
    fn test_from_genome() {
        type C = WConnection;

        let mut inno = InnoGen::new(0);
        let (mut genome, _) = genome::Recurrent::<C>::new(2, 2);
        genome.push_connection(C::new(0, 3, &mut inno));
        genome.push_connection(C::new(0, 1, &mut inno));
        genome.push_connection(C::new(0, 1, &mut inno));

        let nn = Continuous::from_genome(&genome);
        unsafe {
            for c in genome.connections() {
                if c.enabled() {
                    assert_f64_approx!(nn.w.get_unchecked([c.from(), c.to()]), c.weight());
                }
            }

            for (i, node) in genome.nodes().iter().enumerate() {
                assert_f64_approx!(
                    nn.θ.get_unchecked([0, i]),
                    if matches!(node, NodeKind::Static) {
                        1.
                    } else {
                        0.
                    }
                )
            }
        }

        for i in nn.sensory.0..nn.sensory.1 {
            assert!(genome
                .nodes()
                .get(i)
                .is_some_and(|n| matches!(n, NodeKind::Sensory)))
        }
        for i in nn.action.0..nn.action.1 {
            assert!(genome
                .nodes()
                .get(i)
                .is_some_and(|n| matches!(n, NodeKind::Action)))
        }
    }

    /// Tests demonstrating that CTRNNs can produce high confidence outputs (>0.9)
    /// This addresses the concern about whether valid genome configurations exist
    /// that can produce networks capable of high confidence signals.

    #[test]
    fn test_high_confidence_via_strong_bias() {
        // Create a genome with a strong bias-to-output connection
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = genome::Recurrent::<WConnection>::new(1, 1);
        
        // Add strong bias connection
        let mut conn = WConnection::new(2, 1, &mut inno);
        conn.weight = 10.0;
        genome.push_connection(conn);
        
        let mut network = Continuous::from_genome(&genome);
        
        // Test with various inputs
        for input_val in [0.0, 0.5, 1.0] {
            network.flush();
            for _ in 0..50 {
                network.step(10, &[input_val], activate::steep_sigmoid);
            }
            
            let output = network.output()[0];
            assert!(
                output > 0.9,
                "Expected output > 0.9, got {} for input {}",
                output,
                input_val
            );
        }
    }

    #[test]
    fn test_high_confidence_via_self_loop() {
        // Create a genome with input connection and self-reinforcing loop
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = genome::Recurrent::<WConnection>::new(1, 1);
        
        let mut conn1 = WConnection::new(0, 1, &mut inno);
        conn1.weight = 5.0;
        genome.push_connection(conn1);
        
        let mut conn2 = WConnection::new(1, 1, &mut inno);
        conn2.weight = 2.5;
        genome.push_connection(conn2);
        
        let mut network = Continuous::from_genome(&genome);
        
        // Test with high input values
        for input_val in [0.7, 0.8, 0.9, 1.0] {
            network.flush();
            for _ in 0..50 {
                network.step(10, &[input_val], activate::steep_sigmoid);
            }
            
            let output = network.output()[0];
            assert!(
                output > 0.9,
                "Expected output > 0.9, got {} for input {}",
                output,
                input_val
            );
        }
    }

    #[test]
    fn test_high_confidence_controlled_range() {
        // Create a genome with moderate weights for more controlled output
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = genome::Recurrent::<WConnection>::new(1, 1);
        
        let mut conn1 = WConnection::new(0, 1, &mut inno);
        conn1.weight = 2.0;
        genome.push_connection(conn1);
        
        let mut conn2 = WConnection::new(2, 1, &mut inno);
        conn2.weight = 1.5;
        genome.push_connection(conn2);
        
        let mut network = Continuous::from_genome(&genome);
        
        // With fewer steps, output should still exceed 0.9
        network.flush();
        for _ in 0..5 {
            network.step(5, &[1.0], activate::steep_sigmoid);
        }
        
        let output = network.output()[0];
        assert!(
            output > 0.9,
            "Expected output > 0.9, got {}",
            output
        );
    }

    #[test]
    fn test_multiple_outputs_high_confidence() {
        // Create a genome with 2 inputs and 2 outputs
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = genome::Recurrent::<WConnection>::new(2, 2);
        
        // Strong connections from bias to both outputs
        let mut conn1 = WConnection::new(4, 2, &mut inno); // bias to output1
        conn1.weight = 8.0;
        genome.push_connection(conn1);
        
        let mut conn2 = WConnection::new(4, 3, &mut inno); // bias to output2
        conn2.weight = 8.0;
        genome.push_connection(conn2);
        
        let mut network = Continuous::from_genome(&genome);
        
        network.flush();
        for _ in 0..50 {
            network.step(10, &[0.5, 0.5], activate::steep_sigmoid);
        }
        
        let outputs = network.output();
        assert!(
            outputs[0] > 0.9 && outputs[1] > 0.9,
            "Expected both outputs > 0.9, got [{}, {}]",
            outputs[0],
            outputs[1]
        );
    }
}
