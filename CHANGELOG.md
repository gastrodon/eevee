# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-29

### Features

#### Genome Initialization
- **Saturated Initial Connections**: New genomes now create all valid connections between sensory+bias and action neurons, providing a more complete initial topology for evolution

#### Reproduction and Evolution
- **Improved Mutation Error Handling**: Mutations that would fail now return errors instead of panicking, with logic to prefer mutation methods less likely to fail
- **Probabilistic Species Survival**: Weak species now get probabilistic chances to breed based on fractional population allocations, enabling innovation by new or struggling lineages without eliminating them instantly

#### Performance Improvements
- **Hashless Population Allocation**: Optimized species population allocation by removing hash table lookups, using direct iteration over aligned vectors instead
- **FittedGroup Trait**: Introduced `FittedGroup` trait for reasoning about collections of genomes as groups with fitness, eliminating the need for explicit species construction in some cases

#### Code Organization
- **Module Refactoring**: Split reproduction logic into `reproduce.rs` and species/population management into `population.rs` for better code organization

### Improvements

- Added regression tests for population allocation
- Added benchmark for species population allocation
- Minimum fitness for sentiment analysis examples set to > 0
- Fixed and formatted test data for non-bias static nodes
- Build configuration improvements for required features
- XOR example now uses 0-1 gradient fitness calculation for improved training convergence

### Development & CI

- Added code coverage reporting with tarpaulin
- Improved CI workflow with better release candidate detection
- Added codecov artifact generation
- Organized gitignore and improved dependency management
- Removed toml-cli as dev-dependency to avoid conflicts with cargo-tarpaulin

## [0.1.1-1] - 2025-04-11

### Fixed
- Fixed semver versioning (0.1.0-1 is before 0.1.0)
- Added write permission to GITHUB_TOKEN for releases
- Fixed release trigger in CI

## [0.1.0] - 2025-04-10

Initial release with NEAT-based neuroevolution functionality.
