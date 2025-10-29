//! Helpers for de/serializing NeuroEvoluiton components

use crate::{genome::NodeKind, Connection};
use nalgebra as na;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize_matrix<S: Serializer>(
    matrix: &na::DMatrix<f64>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // Convert f64 values to u64 bits for precise serialization
    // Iterate in row-major order to maintain compatibility
    let mut bits = Vec::new();
    for row in matrix.row_iter() {
        for &val in row.iter() {
            bits.push(f64::to_bits(val));
        }
    }

    bits.serialize(serializer)
}

pub fn deserialize_matrix_flat<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<na::DMatrix<f64>, D::Error> {
    Vec::<u64>::deserialize(deserializer).map(|v| {
        // Convert u64 bits back to f64 values
        let float_data: Vec<f64> = v.into_iter().map(f64::from_bits).collect();

        na::DMatrix::from_row_slice(1, float_data.len(), &float_data)
    })
}

pub fn deserialize_matrix_square<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<na::DMatrix<f64>, D::Error> {
    Vec::<u64>::deserialize(deserializer).map(|v| {
        // Convert u64 bits back to f64 values
        let float_data: Vec<f64> = v.into_iter().map(f64::from_bits).collect();

        let n = (float_data.len() as f64).sqrt() as usize;
        debug_assert_eq!(n * n, float_data.len(), "non-square weight vec");
        na::DMatrix::from_row_slice(n, n, &float_data)
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
