//! Centralized constants for Eevee evolution parameters.
//!
//! All configurable parameters are defined here with the `EEVEE_` prefix.
//! This enables easy identification and future environment variable configuration.

use crate::random::percent;

// ============================================================================
// Evolution Parameters
// ============================================================================

/// Number of generations without improvement before species truncation
pub const EEVEE_NO_IMPROVEMENT_TRUNCATE: usize = 10;

// ============================================================================
// Population Parameters
// ============================================================================

/// Genetic distance threshold for speciation
pub const EEVEE_SPECIE_THRESHOLD: f64 = 4.0;

// ============================================================================
// Connection Mutation Parameters
// ============================================================================

/// Probability of disabling a connection during mutation
pub const EEVEE_CONNECTION_DISABLE_PROB: u64 = percent(1);

/// Probability of mutating connection parameters
pub const EEVEE_CONNECTION_MUTATE_PARAM_PROB: u64 = percent(99);

/// Probability of replacing parameter value vs perturbing it
pub const EEVEE_PARAM_REPLACE_PROB: u64 = percent(10);

/// Factor for parameter perturbation (multiplied with random value)
pub const EEVEE_PARAM_PERTURB_FACTOR: f64 = 0.05;

/// Minimum value for parameter mutation range
pub const EEVEE_PARAM_MUTATION_MIN: f64 = -3.0;

/// Maximum value for parameter mutation range
pub const EEVEE_PARAM_MUTATION_MAX: f64 = 3.0;

/// Probability of picking gene from right/less-fit parent in crossover
pub const EEVEE_CROSSOVER_PICK_LESS_FIT_PROB: u64 = percent(50);

/// Probability of keeping disabled genes in crossover
pub const EEVEE_CROSSOVER_KEEP_DISABLED_PROB: u64 = percent(75);

// ============================================================================
// Genome Mutation Parameters
// ============================================================================

/// Probability of mutating individual connections in genome
pub const EEVEE_GENOME_MUTATE_CONNECTION_PROB: u64 = percent(20);

/// Probability of mutating nodes in genome (currently unused)
pub const EEVEE_GENOME_MUTATE_NODE_PROB: u64 = percent(20);

/// Probability of adding a new connection
pub const EEVEE_GENOME_NEW_CONNECTION_PROB: u64 = percent(5);

/// Probability of bisecting a connection (adding a node)
pub const EEVEE_GENOME_BISECT_CONNECTION_PROB: u64 = percent(15);

/// Probability of mutating existing connections
pub const EEVEE_GENOME_MUTATE_EXISTING_PROB: u64 = percent(80);

/// Probability of node mutation in genome event (currently unused)
pub const EEVEE_GENOME_NODE_MUTATION_PROB: u64 = percent(0);

// ============================================================================
// Crossover Coefficients
// ============================================================================

/// Coefficient for excess genes in compatibility distance calculation
pub const EEVEE_CROSSOVER_EXCESS_COEFFICIENT: f64 = 1.0;

/// Coefficient for disjoint genes in compatibility distance calculation
pub const EEVEE_CROSSOVER_DISJOINT_COEFFICIENT: f64 = 1.0;

/// Coefficient for parameter differences in compatibility distance calculation
pub const EEVEE_CROSSOVER_PARAM_COEFFICIENT: f64 = 0.4;

/// Genome size threshold for normalization in delta calculation
pub const EEVEE_CROSSOVER_NORMALIZATION_THRESHOLD: f64 = 20.0;

// ============================================================================
// Reproduction Parameters
// ============================================================================

/// Ratio of offspring from mutation without crossover (1/4 = 0.25)
pub const EEVEE_REPRODUCTION_COPY_RATIO: usize = 4;

/// Number of best individuals to preserve unchanged per generation
pub const EEVEE_REPRODUCTION_CHAMPION_COUNT: usize = 1;
