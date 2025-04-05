#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]
#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

pub mod crossover;
pub mod genome;
pub mod macros;
pub mod network;
pub mod random;
pub mod scenario;
pub mod serialize;
pub mod specie;

pub use genome::{Connection, Genome, Node};
pub use network::{activate, Network};
pub use scenario::{Hook, Scenario, Stats};
pub use specie::Specie;
