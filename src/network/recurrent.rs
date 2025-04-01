use crate::{
    network::serialize::{deserialize_flat, deserialize_square, serialize},
    Network,
};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use serde::{Deserialize, Serialize};

use super::{Recurrent, Stateful};

#[derive(Debug, Serialize, Deserialize)]
pub struct Ctrnn {
    #[serde(serialize_with = "serialize", deserialize_with = "deserialize_flat")]
    pub y: Matrix<f64>, // 1d state of neurons 0-N
    #[serde(serialize_with = "serialize", deserialize_with = "deserialize_flat")]
    pub θ: Matrix<f64>, // 1d bias of neurons 0-N               (\u3b8)
    #[serde(serialize_with = "serialize", deserialize_with = "deserialize_flat")]
    pub τ: Matrix<f64>, // 1d membrane resistance time constant (\u3c4)
    #[serde(serialize_with = "serialize", deserialize_with = "deserialize_square")]
    pub w: Matrix<f64>, // Nd weights between neurons, indexed as [from, to]
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl Network for Ctrnn {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            self.y += (((&self.y + &self.θ).apply(&σ) * &self.w) - &self.y + &m_input)
                .elediv(&self.τ)
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

impl Recurrent for Ctrnn {}

impl Stateful for Ctrnn {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{activate, assert_matrix_approx, random::default_rng};
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

        let original = Ctrnn {
            y: Matrix::new(1, n_neurons, y_data),
            θ: Matrix::new(1, n_neurons, theta_data),
            τ: Matrix::new(1, n_neurons, tau_data),
            w: Matrix::new(n_neurons, n_neurons, w_data),
            sensory: (0, 2),
            action: (3, 5),
        };

        let serialized = original.to_string().expect("Failed to serialize");

        let deserialized = Ctrnn::from_str(&serialized).expect("Failed to deserialize");

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

        let mut original = Ctrnn {
            y: Matrix::new(1, n_neurons, y_data),
            θ: Matrix::new(1, n_neurons, θ_data),
            τ: Matrix::new(1, n_neurons, τ_data),
            w: Matrix::new(n_neurons, n_neurons, w_data),
            sensory: (0, 2),
            action: (3, 5),
        };

        let mut deserialized = Ctrnn::from_str(&original.to_string().expect("Failed to serialize"))
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
}
