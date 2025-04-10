#![allow(confusable_idents)]
#![allow(incomplete_features)]
#![allow(mixed_script_confusables)]

pub mod crossover;
pub mod genome;
pub mod macros;
pub mod network;
pub mod random;
pub mod scenario;
pub mod specie;

pub use genome::{Connection, Genome, Node};
pub use network::{activate, Ctrnn, Network};
pub use random::{Happens, Probabilities};
pub use scenario::{Hook, Scenario, Stats};
pub use specie::Specie;
