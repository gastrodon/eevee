use super::{FromGenome, Recurrent, Stateful};
use crate::{
    genome::NodeKind,
    serialize::{deserialize_matrix_flat, deserialize_matrix_square, serialize_matrix},
    Connection, Genome, Network,
};
use nalgebra as na;
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
    pub y: na::DMatrix<f64>,
    /// 1d bias of neurons 0-N
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub θ: na::DMatrix<f64>,
    /// 1d membrane resistance time constant
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_flat"
    )]
    pub τ: na::DMatrix<f64>,
    /// Nd weights between neurons, indexed as [from, to]
    #[serde(
        serialize_with = "serialize_matrix",
        deserialize_with = "deserialize_matrix_square"
    )]
    pub w: na::DMatrix<f64>,
    /// Range of input neurons, indexing into y
    pub sensory: (usize, usize),
    /// Range of output neurons, indexing into y
    pub action: (usize, usize),
}

impl Network for Continuous {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        let mut m_input = na::DMatrix::zeros(1, self.y.ncols());
        m_input.as_mut_slice()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        
        // Preallocate temporary buffers to reduce allocations
        let mut temp1 = na::DMatrix::zeros(1, self.y.ncols());
        let mut temp2 = na::DMatrix::zeros(1, self.y.ncols());
        
        for _ in 0..prec {
            // temp1 = (y + θ).map(σ)
            temp1.copy_from(&self.y);
            temp1 += &self.θ;
            for val in temp1.iter_mut() {
                *val = σ(*val);
            }
            
            // temp2 = temp1 * w
            temp2.gemm(1.0, &temp1, &self.w, 0.0);
            
            // temp2 = temp2 - y + m_input
            temp2 -= &self.y;
            temp2 += &m_input;
            
            // temp2 = temp2.component_mul(τ) * inv (in-place)
            temp2.component_mul_assign(&self.τ);
            temp2 *= inv;
            
            // y += temp2
            self.y += &temp2;
        }
    }

    fn flush(&mut self) {
        self.y = na::DMatrix::zeros(1, self.y.ncols());
    }

    fn output(&self) -> &[f64] {
        &self.y.as_slice()[self.action.0..self.action.1]
    }
}

impl Recurrent for Continuous {}

impl Stateful for Continuous {}

impl<C: Connection, G: Genome<C>> FromGenome<C, G> for Continuous {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.nodes().len();
        Self {
            y: na::DMatrix::zeros(1, cols),
            θ: na::DMatrix::from_row_slice(
                1,
                cols,
                &genome
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
            τ: na::DMatrix::from_element(1, cols, 0.1),
            w: {
                let mut w = vec![0.; cols * cols];
                for c in genome.connections().iter().filter(|c| c.enabled()) {
                    w[c.from() * cols + c.to()] = c.weight();
                }
                na::DMatrix::from_row_slice(cols, cols, &w)
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
            y: na::DMatrix::from_row_slice(1, n_neurons, &y_data),
            θ: na::DMatrix::from_row_slice(1, n_neurons, &theta_data),
            τ: na::DMatrix::from_row_slice(1, n_neurons, &tau_data),
            w: na::DMatrix::from_row_slice(n_neurons, n_neurons, &w_data),
            sensory: (0, 2),
            action: (3, 5),
        };

        let serialized = original.to_string().expect("Failed to serialize");

        let deserialized = Continuous::from_str(&serialized).expect("Failed to deserialize");

        assert_matrix_approx!(original.y.as_slice(), deserialized.y.as_slice());
        assert_matrix_approx!(original.θ.as_slice(), deserialized.θ.as_slice());
        assert_matrix_approx!(original.τ.as_slice(), deserialized.τ.as_slice());
        assert_matrix_approx!(original.w.as_slice(), deserialized.w.as_slice());

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
            y: na::DMatrix::from_row_slice(1, n_neurons, &y_data),
            θ: na::DMatrix::from_row_slice(1, n_neurons, &θ_data),
            τ: na::DMatrix::from_row_slice(1, n_neurons, &τ_data),
            w: na::DMatrix::from_row_slice(n_neurons, n_neurons, &w_data),
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
        for c in genome.connections() {
            if c.enabled() {
                assert_f64_approx!(nn.w[(c.from(), c.to())], c.weight());
            }
        }

        for (i, node) in genome.nodes().iter().enumerate() {
            assert_f64_approx!(
                nn.θ[(0, i)],
                if matches!(node, NodeKind::Static) {
                    1.
                } else {
                    0.
                }
            )
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
}
