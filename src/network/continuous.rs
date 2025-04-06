use super::{FromGenome, Recurrent, Stateful};
use crate::{
    genome::{Biased, Timescaled},
    serialize::{deserialize_matrix_flat, deserialize_matrix_square, serialize_matrix},
    Connection, Genome, Network, Node,
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
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub y: Matrix<f64>, // 1d state of neurons 0-N
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub θ: Matrix<f64>, // 1d bias of neurons 0-N
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub τ: Matrix<f64>, // 1d membrane resistance time constant
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_square"
    )]
    pub w: Matrix<f64>, // Nd weights between neurons, indexed as [from, to]
    pub sensory: (usize, usize), // Range of input neurons, indexing into y
    pub action: (usize, usize),  // Range of output neurons, indexing into y
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

impl<N: Node + Biased + Timescaled, C: Connection<N>, G: Genome<N, C>> FromGenome<N, C, G>
    for Continuous
{
    fn from_genome(genome: &G) -> Self {
        let cols = genome.nodes().len();
        Self {
            y: Matrix::zeros(1, cols),
            θ: Matrix::new(
                1,
                cols,
                genome.nodes().iter().map(|n| n.bias()).collect::<Vec<_>>(),
            ),
            τ: Matrix::new(
                1,
                cols,
                genome
                    .nodes()
                    .iter()
                    .map(|n| n.timescale())
                    .collect::<Vec<_>>(),
            ),
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
        genome::{self, node::BTNode, NodeKind, WConnection},
        random::default_rng,
        specie::InnoGen,
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
        type N = BTNode;
        type C = WConnection<N>;

        let mut inno = InnoGen::new(0);
        let (mut genome, _) = genome::Recurrent::<N, C>::new(2, 2);
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
                assert_f64_approx!(nn.θ.get_unchecked([0, i]), node.bias())
            }
        }

        for i in nn.sensory.0..nn.sensory.1 {
            assert!(genome
                .nodes()
                .get(i)
                .is_some_and(|n| matches!(n.kind(), NodeKind::Sensory)))
        }
        for i in nn.action.0..nn.action.1 {
            assert!(genome
                .nodes()
                .get(i)
                .is_some_and(|n| matches!(n.kind(), NodeKind::Action)))
        }
    }
}
