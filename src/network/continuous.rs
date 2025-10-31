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
#[derive(Debug)]
pub struct Continuous {
    /// 1d state of neurons 0-N
    pub y: Matrix<f64>,
    /// 1d bias of neurons 0-N
    pub θ: Matrix<f64>,
    /// 1d membrane resistance time constant
    pub τ: Matrix<f64>,
    /// Nd weights between neurons, indexed as [from, to]
    pub w: Matrix<f64>,
    /// Range of input neurons, indexing into y
    pub sensory: (usize, usize),
    /// Range of output neurons, indexing into y
    pub action: (usize, usize),
    /// Reusable buffer for input matrix to avoid allocation in hot loop
    m_input: Matrix<f64>,
}

impl Network for Continuous {
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        // Reset and populate input buffer (reuse allocation)
        self.m_input.mut_data().iter_mut().for_each(|v| *v = 0.);
        self.m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            // Compute full expression to leverage rulinalg's optimizations
            // Breaking it up creates more intermediate allocations than necessary
            self.y += (((&self.y + &self.θ).apply(&σ) * &self.w) - &self.y + &self.m_input)
                .elemul(&self.τ)
                .apply(&|v| v * inv);
        }
    }

    fn flush(&mut self) {
        self.y.mut_data().iter_mut().for_each(|v| *v = 0.);
    }

    #[inline]
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
            m_input: Matrix::zeros(1, cols),
        }
    }
}

// Custom Serialize implementation to skip transient buffers
impl Serialize for Continuous {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Continuous", 6)?;
        
        // Serialize matrix data as u64 bits
        let y_bits: Vec<u64> = self.y.data().iter().map(|&f| f64::to_bits(f)).collect();
        let θ_bits: Vec<u64> = self.θ.data().iter().map(|&f| f64::to_bits(f)).collect();
        let τ_bits: Vec<u64> = self.τ.data().iter().map(|&f| f64::to_bits(f)).collect();
        let w_bits: Vec<u64> = self.w.data().iter().map(|&f| f64::to_bits(f)).collect();
        
        state.serialize_field("y", &y_bits)?;
        state.serialize_field("θ", &θ_bits)?;
        state.serialize_field("τ", &τ_bits)?;
        state.serialize_field("w", &w_bits)?;
        state.serialize_field("sensory", &self.sensory)?;
        state.serialize_field("action", &self.action)?;
        state.end()
    }
}

// Custom Deserialize implementation to initialize transient buffers
impl<'de> Deserialize<'de> for Continuous {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier)]
        enum Field {
            #[serde(rename = "y")]
            Y,
            #[serde(rename = "θ")]
            Θ,
            #[serde(rename = "τ")]
            Τ,
            #[serde(rename = "w")]
            W,
            #[serde(rename = "sensory")]
            Sensory,
            #[serde(rename = "action")]
            Action,
        }

        struct ContinuousVisitor;

        impl<'de> Visitor<'de> for ContinuousVisitor {
            type Value = Continuous;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Continuous")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Continuous, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut y_bits: Option<Vec<u64>> = None;
                let mut θ_bits: Option<Vec<u64>> = None;
                let mut τ_bits: Option<Vec<u64>> = None;
                let mut w_bits: Option<Vec<u64>> = None;
                let mut sensory = None;
                let mut action = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Y => {
                            if y_bits.is_some() {
                                return Err(de::Error::duplicate_field("y"));
                            }
                            y_bits = Some(map.next_value()?);
                        }
                        Field::Θ => {
                            if θ_bits.is_some() {
                                return Err(de::Error::duplicate_field("θ"));
                            }
                            θ_bits = Some(map.next_value()?);
                        }
                        Field::Τ => {
                            if τ_bits.is_some() {
                                return Err(de::Error::duplicate_field("τ"));
                            }
                            τ_bits = Some(map.next_value()?);
                        }
                        Field::W => {
                            if w_bits.is_some() {
                                return Err(de::Error::duplicate_field("w"));
                            }
                            w_bits = Some(map.next_value()?);
                        }
                        Field::Sensory => {
                            if sensory.is_some() {
                                return Err(de::Error::duplicate_field("sensory"));
                            }
                            sensory = Some(map.next_value()?);
                        }
                        Field::Action => {
                            if action.is_some() {
                                return Err(de::Error::duplicate_field("action"));
                            }
                            action = Some(map.next_value()?);
                        }
                    }
                }

                let y_bits = y_bits.ok_or_else(|| de::Error::missing_field("y"))?;
                let θ_bits = θ_bits.ok_or_else(|| de::Error::missing_field("θ"))?;
                let τ_bits = τ_bits.ok_or_else(|| de::Error::missing_field("τ"))?;
                let w_bits = w_bits.ok_or_else(|| de::Error::missing_field("w"))?;
                let sensory = sensory.ok_or_else(|| de::Error::missing_field("sensory"))?;
                let action = action.ok_or_else(|| de::Error::missing_field("action"))?;

                // Convert bits back to f64
                let y_data: Vec<f64> = y_bits.into_iter().map(f64::from_bits).collect();
                let θ_data: Vec<f64> = θ_bits.into_iter().map(f64::from_bits).collect();
                let τ_data: Vec<f64> = τ_bits.into_iter().map(f64::from_bits).collect();
                let w_data: Vec<f64> = w_bits.into_iter().map(f64::from_bits).collect();

                let cols = y_data.len();
                let n = (w_data.len() as f64).sqrt() as usize;
                debug_assert_eq!(n * n, w_data.len(), "non-square weight vec");

                Ok(Continuous {
                    y: Matrix::new(1, cols, y_data),
                    θ: Matrix::new(1, cols, θ_data),
                    τ: Matrix::new(1, cols, τ_data),
                    w: Matrix::new(n, n, w_data),
                    sensory,
                    action,
                    m_input: Matrix::zeros(1, cols),
                })
            }
        }

        const FIELDS: &[&str] = &["y", "θ", "τ", "w", "sensory", "action"];
        deserializer.deserialize_struct("Continuous", FIELDS, ContinuousVisitor)
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
            m_input: Matrix::zeros(1, n_neurons),
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
            m_input: Matrix::zeros(1, n_neurons),
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
}
