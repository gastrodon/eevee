//! Serde Serialize/Deserialize impls for all serializable eevee types.
//! Compiled whenever the `serialize` feature is active (prerequisite for both
//! `serialize_json` and `serialize_binary`).

use rulinalg::matrix::Matrix;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

fn deserialize_connections<'de, C: crate::Connection + Deserialize<'de>, D: Deserializer<'de>>(
    de: D,
) -> Result<Vec<C>, D::Error> {
    Vec::<C>::deserialize(de)
}

/// Generate `Serialize`/`Deserialize` impls for a type via proxy structs, wrapped in a private
/// module named after the type in snake_case.
///
/// Each field may carry `#[serde(...)]` attributes that are forwarded to both the `Ref`
/// (serialize) and `Data` (deserialize) proxy structs.
macro_rules! serde_impl {
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

serde_impl! {
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

serde_impl! {
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

// Simple needs a manual impl because state/sensory/action are derived on deserialize.
mod simple {
    use super::*;
    use crate::network::Simple;

    #[derive(Serialize)]
    struct Ref<'a, C: crate::Connection + Serialize> {
        connections: &'a Vec<C>,
        bias: &'a Vec<f64>,
    }

    #[derive(Deserialize)]
    struct Data<C: crate::Connection + for<'de2> Deserialize<'de2>> {
        #[serde(deserialize_with = "deserialize_connections")]
        connections: Vec<C>,
        bias: Vec<f64>,
    }

    impl<C: crate::Connection + Serialize> Serialize for Simple<C> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            Ref {
                connections: &self.connections,
                bias: &self.bias,
            }
            .serialize(s)
        }
    }

    impl<'de, C: crate::Connection + for<'de2> Deserialize<'de2>> Deserialize<'de> for Simple<C> {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let v = Data::deserialize(d)?;
            let n = v.bias.len();
            Ok(Simple {
                connections: v.connections,
                bias: v.bias,
                state: vec![0.; n],
                sensory: 0..0,
                action: 0..0,
            })
        }
    }
}

serde_impl! {
    use crate::genome::connection::WConnection;

    WConnection {
        inno: usize,
        from: usize,
        to: usize,
        weight: f64,
        enabled: bool,
    }
}

serde_impl! {
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

serde_impl! {
    use crate::genome::recurrent::Recurrent;

    Recurrent<C: crate::Connection> {
        sensory: usize,
        action: usize,
        node_count: usize,
        #[serde(deserialize_with = "deserialize_connections")]
        connections: Vec<C>,
    }
}
