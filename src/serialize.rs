use crate::{genome::NodeKind, Connection};
use rulinalg::matrix::Matrix;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize_matrix<S: Serializer>(
    matrix: &Matrix<f64>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // Convert f64 values to u64 bits for precise serialization
    let bits: Vec<u64> = matrix.data().iter().map(|&f| f64::to_bits(f)).collect();

    bits.serialize(serializer)
}

pub fn deserialize_matrix_flat<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Matrix<f64>, D::Error> {
    Vec::<u64>::deserialize(deserializer).map(|v| {
        // Convert u64 bits back to f64 values
        let float_data: Vec<f64> = v.into_iter().map(f64::from_bits).collect();

        Matrix::new(1, float_data.len(), float_data)
    })
}

pub fn deserialize_matrix_square<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Matrix<f64>, D::Error> {
    Vec::<u64>::deserialize(deserializer).map(|v| {
        // Convert u64 bits back to f64 values
        let float_data: Vec<f64> = v.into_iter().map(f64::from_bits).collect();

        let n = (float_data.len() as f64).sqrt() as usize;
        debug_assert_eq!(n * n, float_data.len(), "non-square weight vec");
        Matrix::new(n, n, float_data)
    })
}

pub fn deserialize_nodes<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<NodeKind>, D::Error> {
    Vec::<NodeKind>::deserialize(deserializer)
}

pub fn deserialize_connections<'de, C: Connection, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<C>, D::Error> {
    Vec::<C>::deserialize(deserializer)
}
