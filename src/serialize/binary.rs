//! Binary serialization via bincode: blanket `SerializeBytes` impl for any `Serialize + Deserialize` type.

use crate::serialize::SerializeBytes;
use serde::{Deserialize, Serialize};

const SERIALIZER_ID: &str = "binary-1";

impl<T: Serialize + for<'de> Deserialize<'de>> SerializeBytes for T {
    const SERIALIZER_ID: &'static str = SERIALIZER_ID;

    fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(bincode::serialize(self)?)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        bincode::deserialize(bytes).map_err(|e| e.into())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        activate, assert_matrix_approx,
        genome::{Recurrent, WConnection},
        network::{Continuous, Network},
        random::default_rng,
        SerializeBytes,
    };
    use rand_distr::{Distribution, Uniform};
    use rulinalg::matrix::Matrix;

    #[test]
    fn test_ctrnn_binary_roundtrip() {
        let n = 10;
        let mut rng = default_rng();
        let dist = Uniform::new(-10f64, 10.).unwrap();

        let mut original = Continuous {
            y: Matrix::new(
                1,
                n,
                (0..n).map(|_| dist.sample(&mut rng)).collect::<Vec<_>>(),
            ),
            θ: Matrix::new(
                1,
                n,
                (0..n).map(|_| dist.sample(&mut rng)).collect::<Vec<_>>(),
            ),
            τ: Matrix::new(
                1,
                n,
                (0..n)
                    .map(|_| dist.sample(&mut rng).abs() + 0.1)
                    .collect::<Vec<_>>(),
            ),
            w: Matrix::new(
                n,
                n,
                (0..n * n)
                    .map(|_| dist.sample(&mut rng))
                    .collect::<Vec<_>>(),
            ),
            sensory: (0, 2),
            action: (3, 5),
        };
        let mut deserialized = Continuous::from_bytes(&original.to_bytes().unwrap()).unwrap();

        for _ in 0..500 {
            let input: Vec<f64> = (0..2).map(|_| dist.sample(&mut rng)).collect();
            original.step(10, &input, activate::steep_sigmoid);
            deserialized.step(10, &input, activate::steep_sigmoid);
            assert_matrix_approx!(original.output(), deserialized.output());
        }
    }

    #[test]
    fn test_genome_binary_roundtrip() {
        use crate::genome::Genome;
        let (genome, _) = Recurrent::<WConnection>::new(2, 1);

        let bytes = genome.to_bytes().unwrap();
        let restored = Recurrent::<WConnection>::from_bytes(&bytes).unwrap();

        assert_eq!(genome.connections().len(), restored.connections().len());
        assert_eq!(genome.sensory(), restored.sensory());
        assert_eq!(genome.action(), restored.action());
    }
}
