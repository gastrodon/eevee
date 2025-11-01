# Context-Based Configuration System
## Environment Variables + RwLock Pattern

**Status:** Design Phase - Documentation  
**Pattern:** Context-like config with multi-reader single-writer  
**Configuration Source:** Environment variables (`EEVEE_*`)  
**Last Updated:** 2025-11-01

---

## Design Overview

### Core Concept

A **context-like configuration struct** that:
1. Gets passed through function calls (like Go's context)
2. Is thread-safe via `Arc<RwLock<Config>>`
3. Loads from environment variables (`EEVEE_*` prefix)
4. Provides standard interface for accessing all configurable parameters
5. Allows many readers, one writer (hot-reloadable config)

---

## Architecture

### Component Diagram

```
Environment Variables
    ↓
Config Loader
    ↓
Arc<RwLock<EeveeConfig>>  ← Context-like struct
    ↓
Passed through function calls
    ↓
Algorithm uses .read() / .write()
```

### Standard Interface

```rust
// src/config/mod.rs

use std::sync::{Arc, RwLock};

/// Configuration context for evolution parameters
/// Thread-safe, sharable, hot-reloadable
#[derive(Clone)]
pub struct EeveeContext {
    inner: Arc<RwLock<EeveeConfig>>,
}

impl EeveeContext {
    /// Create from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let config = EeveeConfig::from_env()?;
        Ok(Self {
            inner: Arc::new(RwLock::new(config)),
        })
    }
    
    /// Create with defaults
    pub fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(EeveeConfig::default())),
        }
    }
    
    /// Read configuration (many readers)
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, EeveeConfig> {
        self.inner.read().unwrap()
    }
    
    /// Write configuration (single writer)
    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, EeveeConfig> {
        self.inner.write().unwrap()
    }
    
    /// Clone the underlying config (for snapshot)
    pub fn snapshot(&self) -> EeveeConfig {
        self.read().clone()
    }
}
```

---

## Complete Configuration Catalog

### All Configurable Constants

Based on comprehensive codebase scan, here are ALL constants that should be configurable:

#### 1. Evolution Parameters (scenario.rs)
```rust
pub struct EvolutionParams {
    /// Generations without improvement before species truncation
    /// Current: const NO_IMPROVEMENT_TRUNCATE: usize = 10
    /// Env: EEVEE_NO_IMPROVEMENT_TRUNCATE
    pub no_improvement_truncate: usize,
}
```

#### 2. Population Parameters (population.rs)
```rust
pub struct PopulationParams {
    /// Genetic distance threshold for speciation
    /// Current: const SPECIE_THRESHOLD: f64 = 4.0
    /// Env: EEVEE_SPECIE_THRESHOLD
    pub specie_threshold: f64,
}
```

#### 3. Connection Mutation Parameters (genome/mod.rs)
```rust
pub struct ConnectionMutationParams {
    /// Probability of disabling vs mutating parameter
    /// Current: const PROBABILITIES: [percent(1), percent(99)]
    /// Env: EEVEE_CONNECTION_DISABLE_PROB, EEVEE_CONNECTION_MUTATE_PROB
    pub disable_probability: f64,
    pub mutate_param_probability: f64,
    
    /// Probability of replacing parameter value vs perturbing
    /// Current: const PARAM_REPLACE_PROBABILITY: u64 = percent(10)
    /// Env: EEVEE_PARAM_REPLACE_PROB
    pub param_replace_probability: f64,
    
    /// Factor for parameter perturbation
    /// Current: const PARAM_PERTURB_FAC: f64 = 0.05
    /// Env: EEVEE_PARAM_PERTURB_FACTOR
    pub param_perturb_factor: f64,
    
    /// Range for parameter mutation sampling
    /// Current: hardcoded -3.0 to 3.0 in macros.rs:102
    /// Env: EEVEE_PARAM_MUTATION_MIN, EEVEE_PARAM_MUTATION_MAX
    pub param_mutation_min: f64,
    pub param_mutation_max: f64,
    
    /// Probability of picking gene from right/less-fit parent
    /// Current: const PROBABILITY_PICK_RL: u64 = percent(50)
    /// Env: EEVEE_CROSSOVER_PICK_LESS_FIT_PROB
    pub probability_pick_less_fit: f64,
    
    /// Probability of keeping disabled genes in crossover
    /// Current: const PROBABILITY_KEEP_DISABLED: u64 = percent(75)
    /// Env: EEVEE_CROSSOVER_KEEP_DISABLED_PROB
    pub probability_keep_disabled: f64,
}
```

#### 4. Genome Mutation Parameters (genome/mod.rs)
```rust
pub struct GenomeMutationParams {
    /// Probability of mutating individual connections
    /// Current: const MUTATE_CONNECTION_PROBABILITY: u64 = percent(20)
    /// Env: EEVEE_GENOME_MUTATE_CONNECTION_PROB
    pub mutate_connection_probability: f64,
    
    /// Probability of mutating nodes (currently unused)
    /// Current: const MUTATE_NODE_PROBABILITY: u64 = percent(20)
    /// Env: EEVEE_GENOME_MUTATE_NODE_PROB
    pub mutate_node_probability: f64,
    
    /// Genome-level event probabilities
    /// Current: const PROBABILITIES: [percent(5), percent(15), percent(80), percent(0)]
    /// Events: NewConnection, BisectConnection, MutateConnection, MutateNode
    /// Env: EEVEE_GENOME_NEW_CONNECTION_PROB, EEVEE_GENOME_BISECT_PROB, 
    ///      EEVEE_GENOME_MUTATE_PROB, EEVEE_GENOME_MUTATE_NODE_PROB
    pub new_connection_probability: f64,
    pub bisect_connection_probability: f64,
    pub mutate_connection_probability: f64,
    pub mutate_node_probability_genome: f64,
}
```

#### 5. Crossover Coefficients (genome/connection.rs, crossover.rs)
```rust
pub struct CrossoverParams {
    /// Coefficient for excess genes in compatibility distance
    /// Current: const EXCESS_COEFFICIENT: f64 = 1.0
    /// Env: EEVEE_EXCESS_COEFFICIENT
    pub excess_coefficient: f64,
    
    /// Coefficient for disjoint genes in compatibility distance
    /// Current: const DISJOINT_COEFFICIENT: f64 = 1.0
    /// Env: EEVEE_DISJOINT_COEFFICIENT
    pub disjoint_coefficient: f64,
    
    /// Coefficient for parameter differences in compatibility distance
    /// Current: const PARAM_COEFFICIENT: f64 = 0.4
    /// Env: EEVEE_PARAM_COEFFICIENT
    pub param_coefficient: f64,
    
    /// Genome size threshold for normalization
    /// Current: hardcoded 20.0 in crossover.rs:125
    /// Env: EEVEE_NORMALIZATION_THRESHOLD
    pub normalization_threshold: f64,
}
```

#### 6. Reproduction Parameters (reproduce.rs)
```rust
pub struct ReproductionParams {
    /// Ratio of offspring from mutation without crossover
    /// Current: hardcoded size / 4 in reproduce.rs:138
    /// Env: EEVEE_REPRODUCTION_COPY_RATIO
    pub copy_ratio: f64,
    
    /// Number of best individuals to preserve unchanged
    /// Current: hardcoded 1 in reproduce.rs:121
    /// Env: EEVEE_CHAMPION_PRESERVATION
    pub champion_preservation: usize,
}
```

#### 7. Network Parameters (network/continuous.rs, network/non_bias.rs)
```rust
pub struct NetworkParams {
    /// Inverse precision for network stepping
    /// Current: hardcoded 1.0 / prec in continuous.rs:53, non_bias.rs:30
    /// Env: EEVEE_NETWORK_PRECISION_INVERSE
    /// Note: This is computed from prec parameter, may not need config
    pub precision_inverse: Option<f64>,
}
```

---

## Environment Variable Specification

### Naming Convention

- Prefix: `EEVEE_`
- Format: `EEVEE_<CATEGORY>_<PARAMETER>`
- Case: UPPER_SNAKE_CASE
- Type annotation in docs

### Complete Environment Variable List

```bash
# Evolution Parameters
EEVEE_NO_IMPROVEMENT_TRUNCATE=10           # usize

# Population Parameters
EEVEE_SPECIE_THRESHOLD=4.0                 # f64

# Connection Mutation
EEVEE_CONNECTION_DISABLE_PROB=0.01         # f64 (0.0-1.0)
EEVEE_CONNECTION_MUTATE_PROB=0.99          # f64 (0.0-1.0)
EEVEE_PARAM_REPLACE_PROB=0.10              # f64 (0.0-1.0)
EEVEE_PARAM_PERTURB_FACTOR=0.05            # f64
EEVEE_PARAM_MUTATION_MIN=-3.0              # f64
EEVEE_PARAM_MUTATION_MAX=3.0               # f64
EEVEE_CROSSOVER_PICK_LESS_FIT_PROB=0.50    # f64 (0.0-1.0)
EEVEE_CROSSOVER_KEEP_DISABLED_PROB=0.75    # f64 (0.0-1.0)

# Genome Mutation
EEVEE_GENOME_MUTATE_CONNECTION_PROB=0.20   # f64 (0.0-1.0)
EEVEE_GENOME_MUTATE_NODE_PROB=0.20         # f64 (0.0-1.0)
EEVEE_GENOME_NEW_CONNECTION_PROB=0.05      # f64 (0.0-1.0)
EEVEE_GENOME_BISECT_PROB=0.15              # f64 (0.0-1.0)
EEVEE_GENOME_MUTATE_PROB=0.80              # f64 (0.0-1.0)

# Crossover Coefficients
EEVEE_EXCESS_COEFFICIENT=1.0               # f64
EEVEE_DISJOINT_COEFFICIENT=1.0             # f64
EEVEE_PARAM_COEFFICIENT=0.4                # f64
EEVEE_NORMALIZATION_THRESHOLD=20.0         # f64

# Reproduction
EEVEE_REPRODUCTION_COPY_RATIO=0.25         # f64 (0.0-1.0)
EEVEE_CHAMPION_PRESERVATION=1              # usize
```

---

## Configuration Module Structure

```rust
// src/config/mod.rs

mod context;
mod params;
mod loader;
mod validation;

pub use context::EeveeContext;
pub use params::{
    EeveeConfig,
    EvolutionParams,
    PopulationParams,
    ConnectionMutationParams,
    GenomeMutationParams,
    CrossoverParams,
    ReproductionParams,
    NetworkParams,
};
pub use loader::ConfigLoader;
pub use validation::ConfigError;
```

---

## Detailed Module Design

### 1. Context Module (context.rs)

```rust
use std::sync::{Arc, RwLock};
use super::{EeveeConfig, ConfigError};

/// Thread-safe configuration context
/// 
/// This struct wraps configuration in Arc<RwLock<>> to enable:
/// - Multiple concurrent readers
/// - Single writer for updates
/// - Cheap cloning (just Arc clone)
/// - Pass-through function calls
#[derive(Clone)]
pub struct EeveeContext {
    inner: Arc<RwLock<EeveeConfig>>,
}

impl EeveeContext {
    /// Create context from environment variables
    /// 
    /// # Example
    /// ```
    /// # use eevee::config::EeveeContext;
    /// let ctx = EeveeContext::from_env().unwrap();
    /// ```
    pub fn from_env() -> Result<Self, ConfigError>;
    
    /// Create context with default configuration
    pub fn default() -> Self;
    
    /// Create context from explicit config
    pub fn new(config: EeveeConfig) -> Self;
    
    /// Get read lock on configuration
    /// Multiple threads can hold read lock simultaneously
    pub fn read(&self) -> RwLockReadGuard<'_, EeveeConfig>;
    
    /// Get write lock on configuration
    /// Only one thread can hold write lock at a time
    pub fn write(&self) -> RwLockWriteGuard<'_, EeveeConfig>;
    
    /// Create snapshot of current configuration
    /// Useful for comparing config changes
    pub fn snapshot(&self) -> EeveeConfig;
    
    /// Update configuration at runtime
    /// 
    /// # Example
    /// ```
    /// # use eevee::config::EeveeContext;
    /// let ctx = EeveeContext::default();
    /// ctx.update(|config| {
    ///     config.population.specie_threshold = 5.0;
    /// });
    /// ```
    pub fn update<F>(&self, f: F) 
    where
        F: FnOnce(&mut EeveeConfig);
    
    /// Reload configuration from environment
    pub fn reload_from_env(&self) -> Result<(), ConfigError>;
}

impl Default for EeveeContext {
    fn default() -> Self {
        Self::new(EeveeConfig::default())
    }
}
```

### 2. Parameters Module (params.rs)

```rust
use serde::{Deserialize, Serialize};

/// Complete configuration for Eevee evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EeveeConfig {
    pub evolution: EvolutionParams,
    pub population: PopulationParams,
    pub connection_mutation: ConnectionMutationParams,
    pub genome_mutation: GenomeMutationParams,
    pub crossover: CrossoverParams,
    pub reproduction: ReproductionParams,
    pub network: NetworkParams,
}

impl Default for EeveeConfig {
    fn default() -> Self {
        Self {
            evolution: EvolutionParams::default(),
            population: PopulationParams::default(),
            connection_mutation: ConnectionMutationParams::default(),
            genome_mutation: GenomeMutationParams::default(),
            crossover: CrossoverParams::default(),
            reproduction: ReproductionParams::default(),
            network: NetworkParams::default(),
        }
    }
}

impl EeveeConfig {
    /// Validate all parameters
    pub fn validate(&self) -> Result<(), ConfigError>;
    
    /// Pre-convert f64 probabilities to u64 for fast comparison
    /// This optimization allows zero-cost abstraction
    pub(crate) fn prepare_for_use(&mut self);
}

// Individual parameter structs with detailed docs...
// (see previous sections for complete definitions)
```

### 3. Loader Module (loader.rs)

```rust
use std::env;
use super::{EeveeConfig, ConfigError};

pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<EeveeConfig, ConfigError> {
        let mut config = EeveeConfig::default();
        
        // Evolution parameters
        if let Some(val) = Self::get_usize("EEVEE_NO_IMPROVEMENT_TRUNCATE")? {
            config.evolution.no_improvement_truncate = val;
        }
        
        // Population parameters
        if let Some(val) = Self::get_f64("EEVEE_SPECIE_THRESHOLD")? {
            config.population.specie_threshold = val;
        }
        
        // Connection mutation
        if let Some(val) = Self::get_f64("EEVEE_CONNECTION_DISABLE_PROB")? {
            config.connection_mutation.disable_probability = val;
        }
        // ... (load all other parameters)
        
        config.validate()?;
        config.prepare_for_use();
        Ok(config)
    }
    
    /// Helper to get optional usize from env
    fn get_usize(key: &str) -> Result<Option<usize>, ConfigError>;
    
    /// Helper to get optional f64 from env
    fn get_f64(key: &str) -> Result<Option<f64>, ConfigError>;
    
    /// Helper to get optional bool from env
    fn get_bool(key: &str) -> Result<Option<bool>, ConfigError>;
    
    /// List all available environment variables
    pub fn list_env_vars() -> Vec<(&'static str, &'static str, &'static str)>;
}
```

### 4. Validation Module (validation.rs)

```rust
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum ConfigError {
    /// Invalid probability (not in [0.0, 1.0])
    InvalidProbability {
        param: String,
        value: f64,
    },
    
    /// Invalid threshold (not positive)
    InvalidThreshold {
        param: String,
        value: f64,
    },
    
    /// Invalid range
    InvalidRange {
        param: String,
        value: f64,
        min: f64,
        max: f64,
    },
    
    /// Environment variable parse error
    EnvParseError {
        var: String,
        value: String,
        expected_type: String,
    },
    
    /// Missing required parameter
    MissingRequired {
        param: String,
    },
}

impl fmt::Display for ConfigError { /* ... */ }
impl Error for ConfigError {}
```

---

## Usage Patterns

### Pattern 1: Load from Environment

```rust
use eevee::config::EeveeContext;

// At program start
let ctx = EeveeContext::from_env()?;

// Pass context to evolve function
evolve_with_context(
    scenario,
    init_fn,
    activation,
    rng,
    hooks,
    ctx.clone(),  // Cheap Arc clone
)?;
```

### Pattern 2: Programmatic Configuration

```rust
let mut ctx = EeveeContext::default();

// Update specific parameters
ctx.update(|config| {
    config.population.specie_threshold = 5.0;
    config.genome_mutation.new_connection_probability = 0.10;
});

evolve_with_context(scenario, init_fn, activation, rng, hooks, ctx)?;
```

### Pattern 3: Hot Reload During Evolution

```rust
// In a separate thread or hook
let ctx_clone = ctx.clone();
std::thread::spawn(move || {
    loop {
        std::thread::sleep(Duration::from_secs(60));
        if let Err(e) = ctx_clone.reload_from_env() {
            eprintln!("Config reload failed: {}", e);
        }
    }
});
```

### Pattern 4: Context-Aware Functions

```rust
// Top level
pub fn evolve_with_context<...>(
    scenario: S,
    init: I,
    σ: A,
    rng: impl RngCore,
    hooks: EvolutionHooks<C, G>,
    ctx: EeveeContext,  // <-- Context passed in
) -> Result<(Vec<Specie<C, G>>, usize), ConfigError> {
    let config = ctx.read();
    
    // Use config.evolution.no_improvement_truncate
    if gen_achieved + config.evolution.no_improvement_truncate <= gen_idx {
        // truncate species
    }
    
    // Pass context down
    population_reproduce(&p_scored, population_lim, inno_head, &mut rng, &ctx)?;
}

// Mid level
fn population_reproduce<C, G>(
    species: &[(Specie<C, G>, f64)],
    population: usize,
    inno_head: usize,
    rng: &mut impl RngCore,
    ctx: &EeveeContext,  // <-- Context passed through
) -> (Vec<G>, usize) {
    let config = ctx.read();
    
    // Use config.reproduction.copy_ratio
    let size_copy = (size as f64 * config.reproduction.copy_ratio) as usize;
    
    // Pass context further
    reproduce(genomes, size, &mut innogen, rng, ctx)?;
}

// Low level
impl Genome<C> for Recurrent<C> {
    fn mutate_with_context(
        &mut self,
        rng: &mut impl RngCore,
        innogen: &mut InnoGen,
        ctx: &EeveeContext,  // <-- Context at leaf level
    ) {
        let config = ctx.read();
        
        // Use config.genome_mutation parameters
        let roll = rng.next_u64();
        let new_conn = config.genome_mutation.new_connection_probability_u64();
        
        if roll < new_conn {
            self.new_connection(rng, innogen);
        }
        // ...
    }
}
```

---

## Performance Optimization

### 1. Pre-converted Probabilities

```rust
pub struct ConnectionMutationParams {
    // Public f64 API
    pub disable_probability: f64,
    
    // Internal u64 cache (computed once)
    disable_probability_u64: u64,
}

impl ConnectionMutationParams {
    pub(crate) fn prepare(&mut self) {
        self.disable_probability_u64 = 
            (self.disable_probability * u64::MAX as f64) as u64;
    }
    
    #[inline(always)]
    pub(crate) fn disable_probability_u64(&self) -> u64 {
        self.disable_probability_u64
    }
}
```

### 2. Lock-Free Fast Path

For read-heavy workloads, consider lock-free alternatives:

```rust
use arc_swap::ArcSwap;

pub struct EeveeContext {
    inner: Arc<ArcSwap<EeveeConfig>>,
}

impl EeveeContext {
    pub fn read(&self) -> arc_swap::Guard<Arc<EeveeConfig>> {
        self.inner.load()  // Lock-free read
    }
}
```

---

## Migration Strategy

### Phase 1: Add Context System (No Breaking Changes)

1. Add `src/config/` module
2. Implement `EeveeContext` and all parameter structs
3. Add environment variable loading
4. Keep all existing APIs unchanged

### Phase 2: Add Context-Aware Functions (Backward Compatible)

1. Add `*_with_context()` variants
2. Old functions call new ones with `EeveeContext::default()`
3. Thread context through call hierarchy

### Phase 3: Deprecate Old APIs (Gradual)

1. Mark old functions as `#[deprecated]`
2. Provide migration guide
3. Eventually remove in next major version

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_config_from_env() {
    env::set_var("EEVEE_SPECIE_THRESHOLD", "5.5");
    let config = ConfigLoader::from_env().unwrap();
    assert_eq!(config.population.specie_threshold, 5.5);
}

#[test]
fn test_context_thread_safety() {
    let ctx = EeveeContext::default();
    let handles: Vec<_> = (0..10).map(|_| {
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let config = ctx.read();
            config.population.specie_threshold
        })
    }).collect();
    
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_hot_reload() {
    let ctx = EeveeContext::default();
    env::set_var("EEVEE_SPECIE_THRESHOLD", "7.0");
    ctx.reload_from_env().unwrap();
    assert_eq!(ctx.read().population.specie_threshold, 7.0);
}
```

---

## Documentation Requirements

### 1. Environment Variable Guide

- Complete list of all `EEVEE_*` variables
- Type information and valid ranges
- Default values
- Examples for different scenarios

### 2. Context API Reference

- Full rustdoc for all public APIs
- Examples for each usage pattern
- Thread safety guarantees
- Performance characteristics

### 3. Migration Guide

- Step-by-step conversion from hardcoded constants
- Before/after code examples
- Performance impact analysis

---

## Future Enhancements

### 1. Config File Support

```rust
impl EeveeContext {
    pub fn from_file(path: &Path) -> Result<Self, ConfigError>;
    pub fn from_toml(contents: &str) -> Result<Self, ConfigError>;
}
```

### 2. Config Validation Levels

```rust
pub enum ValidationLevel {
    Strict,   // Fail on any invalid value
    Lenient,  // Warn and use defaults
    Permissive, // Always succeed with best effort
}
```

### 3. Config Change Notifications

```rust
pub trait ConfigObserver {
    fn on_config_changed(&self, old: &EeveeConfig, new: &EeveeConfig);
}
```

---

## Success Criteria

- [x] Design documented for context-based config
- [ ] All 25+ constants identified and catalogued
- [ ] Environment variable spec complete
- [ ] RwLock pattern designed for thread safety
- [ ] Standard interface defined
- [ ] Migration strategy documented
- [ ] Performance optimization strategy defined
- [ ] Testing strategy defined

---

This design provides a **production-ready configuration system** that:
- ✅ Makes all constants runtime-configurable
- ✅ Uses environment variables for configuration
- ✅ Provides thread-safe multi-reader pattern
- ✅ Enables hot-reloading of config
- ✅ Maintains zero-cost abstraction
- ✅ Backward compatible with existing API
