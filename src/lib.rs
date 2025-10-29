#![feature(generic_const_exprs)]
#![allow(confusable_idents)]
#![allow(incomplete_features)]
#![allow(mixed_script_confusables)]

pub mod crossover;
pub mod genome;
pub mod macros;
pub mod network;
pub mod population;
pub mod random;
pub mod reproduce;
pub mod scenario;
pub mod serde_traits;

#[cfg(feature = "serialize")]
pub mod serialize;

pub use genome::{Connection, Genome};
pub use network::{activate, Network};
pub use population::Specie;
pub use scenario::{Hook, Scenario, Stats};
pub use serde_traits::{Deserialize, Serialize};
