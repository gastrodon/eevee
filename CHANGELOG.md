# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-05-16

### Features

Serialization
--------------
- **Optional Serialization Features `d9b951e`**: Serde support now gated behind optional `serialize` and `serialize_json` features, reducing dependency bloat for users who don't need serialization
- **Native Nalgebra Serialization `b9b2153`**: Bit-exact f64 encoding for nalgebra types with proper serde integration

### Changed

Core Refactoring
----------------
- **Matrix Library Migration `e69fa75`**: Migrated from rulinalg to nalgebra for all matrix operations, improving performance and reducing dependency complexity
- **Removed Static Bias Node `4a9d8a3`**: Simplified genome and network architecture by removing the static bias node; bias is now only dynamically added when needed
- **Removed NodeKind Enum `231644b`**: Replaced discriminated `NodeKind` enum with simple `node_count: usize` for cleaner architecture

Test Infrastructure
-------------------
- **Parametric Test Macro `8b340df`**: Introduced `fn_matrix!` proc macro for composable parametric testing across network types and configurations
- **Comprehensive Network Tests `c360b50`**: Added extensive trait tests for network implementations with network-specific test modules

Tooling & Build
---------------
- **Extended Profiling Tools `8fcf98b`**: Added `toml2json` utility and `--all` flag to profiling scripts for better benchmarking analysis

### Examples

- Removed sentiment analysis example in favor of XOR evolution benchmark `151c238`
- Extracted XOR scenario into dedicated module for better organization `e4ae5d1`
- XOR benchmark now focuses purely on evolution dynamics with randomized training pairs `ba98125`

## [0.2.0] - 2025-10-29

### Features

Genome Initialization
---------------------
- **Saturated Initial Connections `28db946`**: New genomes now create all valid connections between sensory+bias and action neurons, providing a more complete initial topology for evolution

Reproduction and Evolution
--------------------------
- **Improved Mutation Error Handling `65bfdf0`**: Mutations that would fail now return errors instead of panicking, with logic to prefer mutation methods less likely to fail
- **Removed Selective Reproduction `b659314`**: Simplified reproduction pipeline to accept all species members, eliminating fitness threshold filtering while maintaining proportional allocation based on adjusted fitness

Performance Improvements
------------------------
- **Hashless Population Allocation `d3abc26`**: Optimized species population allocation by removing hash table lookups, using direct iteration over aligned vectors instead
- **FittedGroup Trait `7bd4da6`**: Introduced `FittedGroup` trait for reasoning about collections of genomes as groups with fitness, eliminating the need for explicit species construction in some cases

Code Organization
-----------------
- **Module Refactoring `f6e3a99`**: Split reproduction logic into `reproduce.rs` and species/population management into `population.rs` for better code organization

### Improvements

#### Testing
- Added regression tests for population allocation `155487a`
- Fixed and formatted test data `a41a750`
- Fixed test data for non-bias static nodes `d29288b`

#### Benchmarks
- Added benchmark for species population allocation `9eb974b`

#### Examples
- Minimum fitness for sentiment analysis examples set to > 0 `6caddab`
- XOR example now uses 0-1 gradient fitness calculation for improved training convergence `1fc9556`

#### Build Configuration
- Build configuration improvements for required features `f12687c`
- Renamed package in build config `41085ed`
- Sorted dependencies and hoisted dev-dependencies for better organization `03dba6a`, `90435dc`
- Pinned dependencies to minor versions `4ab9276`
- Explicitly prefixed trait dependencies `b519da9`

### Development & CI

#### Code Coverage
- Added code coverage reporting with tarpaulin `5dd8161`
- Added codecov artifact generation `1978881`
- Removed toml-cli as dev-dependency to avoid conflicts with cargo-tarpaulin `d216ae6`

#### CI Workflow Improvements
- Improved CI workflow with better release candidate detection `8d62067`
- Added separate dry-run publish for RC tags `0a3f27e`
- Post-test steps now require build and test completion `f256e65`
- Configured toolchain manually `3164615`
- Test that all features build `2602ab6`

#### Project Organization
- Organized gitignore `31ac7fd`

## [0.1.1-1] - 2025-04-11

### Fixed
- Fixed semver versioning (0.1.0-1 is before 0.1.0)
- Added write permission to GITHUB_TOKEN for releases
- Fixed release trigger in CI

## [0.1.0] - 2025-04-10

Initial release with NEAT-based neuroevolution functionality.
