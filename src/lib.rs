#![feature(generic_const_exprs)]
#![allow(confusable_idents)]
#![allow(incomplete_features)]
#![allow(mixed_script_confusables)]
//
#![feature(custom_test_frameworks)]
#![cfg_attr(not(feature = "bench"), test_runner(runner))]
#![cfg_attr(feature = "bench", test_runner(criterion::runner))]

#[doc(hidden)]
pub fn runner(tests: &[&dyn Fn()]) {
    let mut i = 0;
    for test in tests {
        test();
        i += 1;
    }
    println!("all {i} tests passed")
}

pub mod crossover;
pub mod genome;
pub mod macros;
pub mod network;
pub mod population;
pub mod random;
pub mod reproduce;
pub mod scenario;
pub mod serialize;

pub use genome::{Connection, Genome};
pub use network::{activate, Network};
pub use population::Specie;
pub use scenario::{Hook, Scenario, Stats};
