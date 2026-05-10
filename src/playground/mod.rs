//! Example scenarios for evaluating neuroevolution genomes.
//!
//! This module provides concrete test cases (scenarios) that can be used to
//! benchmark and validate the NEAT algorithm. Each scenario implements the
//! [`Scenario`] trait with different evaluation logic.
//!
//! # Examples
//!
//! - [`xor::XorScenario`] — Binary classification on XOR pairs

pub mod xor;
