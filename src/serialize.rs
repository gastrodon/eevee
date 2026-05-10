//! Helpers for de/serializing NeuroEvolution components

/// Serde module for `DMatrix<f64>` using nalgebra's `(data, nrows, ncols)` tuple
/// format with u64 bit-encoding for exact f64 round-trip.
#[cfg(feature = "serialize")]
pub mod dmatrix {
    use nalgebra as na;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(m: &na::DMatrix<f64>, s: S) -> Result<S::Ok, S::Error> {
        let bits: Vec<u64> = m.as_slice().iter().map(|&f| f.to_bits()).collect();
        (bits, m.nrows(), m.ncols()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<na::DMatrix<f64>, D::Error> {
        let (bits, nrows, ncols): (Vec<u64>, usize, usize) = Deserialize::deserialize(d)?;
        let data: Vec<f64> = bits.into_iter().map(f64::from_bits).collect();
        Ok(na::DMatrix::from_vec(nrows, ncols, data))
    }
}
