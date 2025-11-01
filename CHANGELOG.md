# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-29

### Features

#### Genome Initialization
- **Saturated Initial Connections**: New genomes now create all valid connections between sensory+bias and action neurons, providing a more complete initial topology for evolution ([`28db946`](https://github.com/gastrodon/eevee/commit/28db946))

#### Reproduction and Evolution
- **Improved Mutation Error Handling**: Mutations that would fail now return errors instead of panicking, with logic to prefer mutation methods less likely to fail ([`65bfdf0`](https://github.com/gastrodon/eevee/commit/65bfdf0))
- **Removed Selective Reproduction**: Simplified reproduction pipeline to accept all species members, eliminating fitness threshold filtering while maintaining proportional allocation based on adjusted fitness ([`b659314`](https://github.com/gastrodon/eevee/commit/b659314))

#### Performance Improvements
- **Hashless Population Allocation**: Optimized species population allocation by removing hash table lookups, using direct iteration over aligned vectors instead ([`d3abc26`](https://github.com/gastrodon/eevee/commit/d3abc26))
- **FittedGroup Trait**: Introduced `FittedGroup` trait for reasoning about collections of genomes as groups with fitness, eliminating the need for explicit species construction in some cases ([`7bd4da6`](https://github.com/gastrodon/eevee/commit/7bd4da6))

#### Code Organization
- **Module Refactoring**: Split reproduction logic into `reproduce.rs` and species/population management into `population.rs` for better code organization ([`f6e3a99`](https://github.com/gastrodon/eevee/commit/f6e3a99))

### Improvements

#### Testing
- Added regression tests for population allocation ([`155487a`](https://github.com/gastrodon/eevee/commit/155487a))
- Fixed and formatted test data ([`a41a750`](https://github.com/gastrodon/eevee/commit/a41a750))
- Fixed test data for non-bias static nodes ([`d29288b`](https://github.com/gastrodon/eevee/commit/d29288b))

#### Benchmarks
- Added benchmark for species population allocation ([`9eb974b`](https://github.com/gastrodon/eevee/commit/9eb974b))

#### Examples
- Minimum fitness for sentiment analysis examples set to > 0 ([`6caddab`](https://github.com/gastrodon/eevee/commit/6caddab))
- XOR example now uses 0-1 gradient fitness calculation for improved training convergence ([`1fc9556`](https://github.com/gastrodon/eevee/commit/1fc9556))

#### Build Configuration
- Build configuration improvements for required features ([`f12687c`](https://github.com/gastrodon/eevee/commit/f12687c))
- Renamed package in build config ([`41085ed`](https://github.com/gastrodon/eevee/commit/41085ed))
- Sorted dependencies and hoisted dev-dependencies for better organization ([`03dba6a`](https://github.com/gastrodon/eevee/commit/03dba6a), [`90435dc`](https://github.com/gastrodon/eevee/commit/90435dc))
- Pinned dependencies to minor versions ([`4ab9276`](https://github.com/gastrodon/eevee/commit/4ab9276))
- Explicitly prefixed trait dependencies ([`b519da9`](https://github.com/gastrodon/eevee/commit/b519da9))

### Development & CI

#### Code Coverage
- Added code coverage reporting with tarpaulin ([`5dd8161`](https://github.com/gastrodon/eevee/commit/5dd8161))
- Added codecov artifact generation ([`1978881`](https://github.com/gastrodon/eevee/commit/1978881))
- Removed toml-cli as dev-dependency to avoid conflicts with cargo-tarpaulin ([`d216ae6`](https://github.com/gastrodon/eevee/commit/d216ae6))

#### CI Workflow Improvements
- Improved CI workflow with better release candidate detection ([`8d62067`](https://github.com/gastrodon/eevee/commit/8d62067))
- Added separate dry-run publish for RC tags ([`0a3f27e`](https://github.com/gastrodon/eevee/commit/0a3f27e))
- Post-test steps now require build and test completion ([`f256e65`](https://github.com/gastrodon/eevee/commit/f256e65))
- Configured toolchain manually ([`3164615`](https://github.com/gastrodon/eevee/commit/3164615))
- Test that all features build ([`2602ab6`](https://github.com/gastrodon/eevee/commit/2602ab6))

#### Project Organization
- Organized gitignore ([`31ac7fd`](https://github.com/gastrodon/eevee/commit/31ac7fd))

## [0.1.1-1] - 2025-04-11

### Fixed
- Fixed semver versioning (0.1.0-1 is before 0.1.0)
- Added write permission to GITHUB_TOKEN for releases
- Fixed release trigger in CI

## [0.1.0] - 2025-04-10

Initial release with NEAT-based neuroevolution functionality.
