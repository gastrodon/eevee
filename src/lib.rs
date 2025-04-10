#![feature(generic_const_exprs)]
#![allow(confusable_idents)]
#![allow(incomplete_features)]
#![allow(mixed_script_confusables)]

pub mod crossover;
pub mod genome;
pub mod macros;
pub mod network;
pub mod random;
pub mod scenario;
pub mod serialize;
pub mod specie;

pub use genome::{Connection, Genome};
pub use network::{activate, Network};
pub use scenario::{Hook, Scenario, Stats};
pub use specie::Specie;
