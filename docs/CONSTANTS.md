# Centralized Constants Reference
## EEVEE_* Configuration Parameters

This document catalogs all centralized constants in the Eevee library. All constants follow the `EEVEE_*` naming convention for easy identification and future environment variable configuration.

---

## Location

All constants are defined in: `src/constants.rs`

## Constant Categories

### Evolution Parameters

| Constant | Value | Description | Location |
|----------|-------|-------------|----------|
| `EEVEE_NO_IMPROVEMENT_TRUNCATE` | `10` | Generations without improvement before species truncation | `scenario.rs` |

### Population Parameters

| Constant | Value | Description | Location |
|----------|-------|-------------|----------|
| `EEVEE_SPECIE_THRESHOLD` | `4.0` | Genetic distance threshold for speciation | `population.rs` |

### Connection Mutation Parameters

| Constant | Value | Description | Location |
|----------|-------|-------------|----------|
| `EEVEE_CONNECTION_DISABLE_PROB` | `percent(1)` | Probability of disabling a connection | `genome/mod.rs` trait |
| `EEVEE_CONNECTION_MUTATE_PARAM_PROB` | `percent(99)` | Probability of mutating connection parameters | `genome/mod.rs` trait |
| `EEVEE_PARAM_REPLACE_PROB` | `percent(10)` | Probability of replacing vs perturbing parameter | `genome/mod.rs` trait |
| `EEVEE_PARAM_PERTURB_FACTOR` | `0.05` | Factor for parameter perturbation | `genome/mod.rs` trait |
| `EEVEE_PARAM_MUTATION_MIN` | `-3.0` | Minimum value for parameter mutation range | `macros.rs` |
| `EEVEE_PARAM_MUTATION_MAX` | `3.0` | Maximum value for parameter mutation range | `macros.rs` |
| `EEVEE_CROSSOVER_PICK_LESS_FIT_PROB` | `percent(50)` | Probability of picking gene from less fit parent | `genome/mod.rs` trait |
| `EEVEE_CROSSOVER_KEEP_DISABLED_PROB` | `percent(75)` | Probability of keeping disabled genes | `genome/mod.rs` trait |

### Genome Mutation Parameters

| Constant | Value | Description | Location |
|----------|-------|-------------|----------|
| `EEVEE_GENOME_MUTATE_CONNECTION_PROB` | `percent(20)` | Probability of mutating individual connections | `genome/mod.rs` trait |
| `EEVEE_GENOME_MUTATE_NODE_PROB` | `percent(20)` | Probability of mutating nodes (unused) | `genome/mod.rs` trait |
| `EEVEE_GENOME_NEW_CONNECTION_PROB` | `percent(5)` | Probability of adding new connection | `genome/mod.rs` trait |
| `EEVEE_GENOME_BISECT_CONNECTION_PROB` | `percent(15)` | Probability of bisecting connection | `genome/mod.rs` trait |
| `EEVEE_GENOME_MUTATE_EXISTING_PROB` | `percent(80)` | Probability of mutating existing connections | `genome/mod.rs` trait |
| `EEVEE_GENOME_NODE_MUTATION_PROB` | `percent(0)` | Probability of node mutation (unused) | `genome/mod.rs` trait |

### Crossover Coefficients

| Constant | Value | Description | Location |
|----------|-------|-------------|----------|
| `EEVEE_CROSSOVER_EXCESS_COEFFICIENT` | `1.0` | Coefficient for excess genes in compatibility | `genome/connection.rs` |
| `EEVEE_CROSSOVER_DISJOINT_COEFFICIENT` | `1.0` | Coefficient for disjoint genes in compatibility | `genome/connection.rs` |
| `EEVEE_CROSSOVER_PARAM_COEFFICIENT` | `0.4` | Coefficient for parameter differences | `genome/connection.rs` |
| `EEVEE_CROSSOVER_NORMALIZATION_THRESHOLD` | `20.0` | Genome size threshold for normalization | `crossover.rs` |

### Reproduction Parameters

| Constant | Value | Description | Location |
|----------|-------|-------------|----------|
| `EEVEE_REPRODUCTION_COPY_RATIO` | `4` | Ratio for offspring from mutation (1/4 = 25%) | `reproduce.rs` |
| `EEVEE_REPRODUCTION_CHAMPION_COUNT` | `1` | Number of champions to preserve | `reproduce.rs` |

---

## Usage

Constants are used throughout the codebase via the `crate::constants::` module:

```rust
use crate::constants::EEVEE_SPECIE_THRESHOLD;

if repr.delta(genome.connections()) < EEVEE_SPECIE_THRESHOLD {
    // genome belongs to this species
}
```

---

## Benefits of Centralization

1. **Single Source of Truth**: All configurable values in one location
2. **Easy Identification**: `EEVEE_*` prefix makes constants obvious
3. **Future-Ready**: Prepared for environment variable configuration
4. **Consistent Naming**: Related parameters grouped with similar prefixes
5. **Documentation**: Clear descriptions and purpose for each constant

---

## Migration from Hardcoded Values

All previously hardcoded values have been migrated:

- ✅ Trait constants now reference `constants::`
- ✅ Magic numbers replaced with named constants
- ✅ Inline literals extracted to centralized location
- ✅ Consistent naming convention applied

### Example Migration

**Before:**
```rust
const SPECIE_THRESHOLD: f64 = 4.;  // in population.rs
const NO_IMPROVEMENT_TRUNCATE: usize = 10;  // in scenario.rs
```

**After:**
```rust
// All in src/constants.rs
pub const EEVEE_SPECIE_THRESHOLD: f64 = 4.0;
pub const EEVEE_NO_IMPROVEMENT_TRUNCATE: usize = 10;
```

---

## Future Enhancements

This centralization prepares for:

1. **Environment Variable Configuration**: Load from `EEVEE_*` env vars
2. **Runtime Configuration**: Context-based parameter adjustment
3. **Configuration Files**: TOML/YAML/JSON configuration support
4. **Preset Profiles**: Domain-specific parameter sets
5. **Hot Reloading**: Update parameters without restart

See `docs/design/context_config_system.md` for the full configuration system design.

---

## Verification

All constants tested and verified:
- ✅ Library builds successfully
- ✅ All 64 tests pass
- ✅ Backward compatible with existing code
- ✅ No behavioral changes, only refactoring

---

Last Updated: 2025-11-03
