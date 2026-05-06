//! JSON serialization: blanket `SerializeFile` impl, field helpers, and per-type impls.

use crate::serialize::SerializeFile;
use crate::Connection;
use rulinalg::matrix::Matrix;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::Range;

const SERIALIZER_ID: &str = "json-1";

impl<T: Serialize + for<'de> Deserialize<'de>> SerializeFile for T {
    const SERIALIZER_ID: &'static str = SERIALIZER_ID;

    fn to_str(&self) -> Result<String, Box<dyn std::error::Error>> {
        Ok(serde_json::to_string(self)?)
    }

    fn from_str(s: &str) -> Result<Self, Box<dyn std::error::Error>> {
        serde_json::from_str(s).map_err(|e| e.into())
    }
}

fn serialize_matrix<S: Serializer>(matrix: &Matrix<f64>, ser: S) -> Result<S::Ok, S::Error> {
    let bits: Vec<u64> = matrix.data().iter().map(|&f| f64::to_bits(f)).collect();
    bits.serialize(ser)
}

fn deserialize_matrix_flat<'de, D: Deserializer<'de>>(de: D) -> Result<Matrix<f64>, D::Error> {
    Vec::<u64>::deserialize(de).map(|v| {
        let float_data: Vec<f64> = v.into_iter().map(f64::from_bits).collect();
        Matrix::new(1, float_data.len(), float_data)
    })
}

fn deserialize_matrix_square<'de, D: Deserializer<'de>>(de: D) -> Result<Matrix<f64>, D::Error> {
    Vec::<u64>::deserialize(de).map(|v| {
        let float_data: Vec<f64> = v.into_iter().map(f64::from_bits).collect();
        let n = (float_data.len() as f64).sqrt() as usize;
        debug_assert_eq!(n * n, float_data.len(), "non-square weight vec");
        Matrix::new(n, n, float_data)
    })
}

fn deserialize_connections<'de, C: Connection + Deserialize<'de>, D: Deserializer<'de>>(
    de: D,
) -> Result<Vec<C>, D::Error> {
    Vec::<C>::deserialize(de)
}

/// Generate `Serialize`/`Deserialize` impls for a type, wrapped in a private module named
/// after the type in snake_case.
///
/// Each field may carry any number of `#[serde(...)]` attributes placed on both the `Ref`
/// (serialize) and `Data` (deserialize) proxy structs.  Serde silently ignores attributes
/// that don't apply to the current derive direction, so combining `serialize_with` and
/// `deserialize_with` on the same field is safe.
///
/// Optional `#[...]` attributes before the `use` are forwarded to the generated module
/// (useful for `#[allow(deprecated)]` etc.).
///
/// # Non-generic
/// ```ignore
/// json_impl! {
///     use crate::network::Continuous;
///     Continuous {
///         plain_field: u32,
///         #[serde(serialize_with = "ser_fn", deserialize_with = "de_fn")]
///         matrix_field: Matrix<f64>,
///     }
/// }
/// ```
///
/// # Single generic type parameter
/// ```ignore
/// json_impl! {
///     use crate::genome::Recurrent;
///     Recurrent<C: Connection> { items: Vec<C> }
/// }
/// ```
/// The macro adds `+ Serialize` for the `Serialize` impl and `+ for<'de2> Deserialize<'de2>`
/// for the `Deserialize` impl automatically.
macro_rules! json_impl {
    // Non-generic
    (
        $(#[$mod_attr:meta])*
        use $use_path:path;
        $Type:ident {
            $($(#[$attr:meta])* $field:ident : $ftype:ty),* $(,)?
        }
        $($extra:item)*
    ) => {
        ::paste::paste! {
            $(#[$mod_attr])*
            mod [<$Type:snake>] {
                use super::*;
                use $use_path;

                #[derive(Serialize)]
                struct Ref<'a> {
                    $($(#[$attr])* $field: &'a $ftype,)*
                }

                #[derive(Deserialize)]
                struct Data {
                    $($(#[$attr])* $field: $ftype,)*
                }

                impl Serialize for $Type {
                    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                        Ref { $($field: &self.$field,)* }.serialize(s)
                    }
                }

                impl<'de> Deserialize<'de> for $Type {
                    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                        let v = Data::deserialize(d)?;
                        Ok($Type { $($field: v.$field,)* })
                    }
                }

                $($extra)*
            }
        }
    };

    // Single generic type parameter with a single base trait bound
    (
        $(#[$mod_attr:meta])*
        use $use_path:path;
        $Type:ident < $GP:ident : $Bound:path > {
            $($(#[$attr:meta])* $field:ident : $ftype:ty),* $(,)?
        }
        $($extra:item)*
    ) => {
        ::paste::paste! {
            $(#[$mod_attr])*
            mod [<$Type:snake>] {
                use super::*;
                use $use_path;

                #[derive(Serialize)]
                struct Ref<'a, $GP: $Bound + Serialize> {
                    $($(#[$attr])* $field: &'a $ftype,)*
                }

                #[derive(Deserialize)]
                struct Data<$GP: $Bound + for<'de2> Deserialize<'de2>> {
                    $($(#[$attr])* $field: $ftype,)*
                }

                impl<$GP: $Bound + Serialize> Serialize for $Type<$GP> {
                    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                        Ref { $($field: &self.$field,)* }.serialize(s)
                    }
                }

                impl<'de, $GP: $Bound + for<'de2> Deserialize<'de2>> Deserialize<'de>
                    for $Type<$GP>
                {
                    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                        let v = Data::deserialize(d)?;
                        Ok($Type { $($field: v.$field,)* })
                    }
                }

                $($extra)*
            }
        }
    };
}

json_impl! {
    use crate::network::Continuous;

    Continuous {
        #[serde(serialize_with = "serialize_matrix", deserialize_with = "deserialize_matrix_flat")]
        y: Matrix<f64>,
        #[serde(serialize_with = "serialize_matrix", deserialize_with = "deserialize_matrix_flat")]
        θ: Matrix<f64>,
        #[serde(serialize_with = "serialize_matrix", deserialize_with = "deserialize_matrix_flat")]
        τ: Matrix<f64>,
        #[serde(serialize_with = "serialize_matrix", deserialize_with = "deserialize_matrix_square")]
        w: Matrix<f64>,
        sensory: (usize, usize),
        action: (usize, usize),
    }
}

json_impl! {
    use crate::network::NonBias;

    NonBias {
        #[serde(serialize_with = "serialize_matrix", deserialize_with = "deserialize_matrix_flat")]
        y: Matrix<f64>,
        #[serde(serialize_with = "serialize_matrix", deserialize_with = "deserialize_matrix_square")]
        w: Matrix<f64>,
        sensory: (usize, usize),
        action: (usize, usize),
    }
}

json_impl! {
    use crate::network::Simple;

    Simple<C: Connection> {
        #[serde(deserialize_with = "deserialize_connections")]
        connections: Vec<C>,
        bias: Vec<f64>,
        state: Vec<f64>,
        sensory: Range<usize>,
        action: Range<usize>,
    }
}

json_impl! {
    use crate::genome::Recurrent;

    Recurrent<C: Connection> {
        sensory: usize,
        action: usize,
        node_count: usize,
        #[serde(deserialize_with = "deserialize_connections")]
        connections: Vec<C>,
    }
}

json_impl! {
    use crate::genome::connection::WConnection;

    WConnection {
        inno: usize,
        from: usize,
        to: usize,
        weight: f64,
        enabled: bool,
    }
}

json_impl! {
    use crate::genome::connection::BWConnection;

    BWConnection {
        inno: usize,
        from: usize,
        to: usize,
        bias: f64,
        weight: f64,
        enabled: bool,
    }
}

#[cfg(test)]
mod test {
    use crate::{
        activate, assert_matrix_approx,
        genome::{Recurrent as RecurrentGenome, WConnection},
        network::{Continuous, FromGenome, Network, NonBias, Simple},
        random::default_rng,
        Genome, SerializeFile,
    };
    use rand_distr::{Distribution, Uniform};
    use rulinalg::matrix::Matrix;

    fn rand_square(n: usize, dist: &Uniform<f64>, rng: &mut impl rand::RngCore) -> Matrix<f64> {
        Matrix::new(
            n,
            n,
            (0..n * n).map(|_| dist.sample(rng)).collect::<Vec<_>>(),
        )
    }

    fn rand_row(n: usize, dist: &Uniform<f64>, rng: &mut impl rand::RngCore) -> Matrix<f64> {
        Matrix::new(1, n, (0..n).map(|_| dist.sample(rng)).collect::<Vec<_>>())
    }

    #[test]
    fn test_ctrnn_behavioral_equivalence() {
        let n = 10;
        let mut rng = default_rng();
        let dist = Uniform::new(-10f64, 10.).unwrap();

        let mut original = Continuous {
            y: rand_row(n, &dist, &mut rng),
            θ: rand_row(n, &dist, &mut rng),
            τ: Matrix::new(
                1,
                n,
                (0..n)
                    .map(|_| dist.sample(&mut rng).abs() + 0.1)
                    .collect::<Vec<_>>(),
            ),
            w: rand_square(n, &dist, &mut rng),
            sensory: (0, 2),
            action: (3, 5),
        };
        let mut deserialized = Continuous::from_str(&original.to_str().unwrap()).unwrap();

        for _ in 0..500 {
            let input: Vec<f64> = (0..2).map(|_| dist.sample(&mut rng)).collect();
            original.step(10, &input, activate::steep_sigmoid);
            deserialized.step(10, &input, activate::steep_sigmoid);
            assert_matrix_approx!(original.output(), deserialized.output());
        }
    }

    #[test]
    fn test_non_bias_behavioral_equivalence() {
        let n = 8;
        let mut rng = default_rng();
        let dist = Uniform::new(-5f64, 5.).unwrap();

        let mut original = NonBias {
            y: rand_row(n, &dist, &mut rng),
            w: rand_square(n, &dist, &mut rng),
            sensory: (0, 2),
            action: (4, 6),
        };
        let mut deserialized = NonBias::from_str(&original.to_str().unwrap()).unwrap();

        for _ in 0..500 {
            let input: Vec<f64> = (0..2).map(|_| dist.sample(&mut rng)).collect();
            original.step(10, &input, activate::steep_sigmoid);
            deserialized.step(10, &input, activate::steep_sigmoid);
            assert_matrix_approx!(original.output(), deserialized.output());
        }
    }

    #[test]
    fn test_simple_behavioral_equivalence() {
        let dist = Uniform::new(-5f64, 5.).unwrap();
        let mut rng = default_rng();

        let (genome, _) = RecurrentGenome::<WConnection>::new(3, 2);
        let mut original = Simple::from_genome(&genome);
        let mut deserialized =
            Simple::<WConnection>::from_str(&original.to_str().unwrap()).unwrap();

        for _ in 0..500 {
            let input: Vec<f64> = (0..3).map(|_| dist.sample(&mut rng)).collect();
            original.step(5, &input, activate::steep_sigmoid);
            deserialized.step(5, &input, activate::steep_sigmoid);
            assert_matrix_approx!(original.output(), deserialized.output());
        }
    }
}
