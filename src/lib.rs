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
#[cfg(feature = "serialize")]
pub mod serialize;

pub use genome::{Connection, Genome};
pub use network::{activate, Network, ToNetwork};
pub use population::Specie;
pub use scenario::{EvolutionConfig, Hook, Scenario, Stats};
#[cfg(feature = "serialize")]
pub use serialize::SerializeFile;

/// Topology marker: network or genome allows recurrent (cyclic) connections.
pub trait Recurrent {}

/// Topology marker: network or genome guarantees acyclic (feedforward) connections.
pub trait Forward {}
