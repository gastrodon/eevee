# Implementation Guide: Configuration System
## Priority 1 - Foundation for Parameterizable Library

---

## Overview

This guide provides a concrete implementation plan for the most critical quality of life feature: a runtime configuration system. This addresses the core limitation that all parameters are currently hardcoded.

---

## Architecture Design

### Core Configuration Structs

```rust
// src/config/mod.rs
pub mod evolution;
pub mod mutation;
pub mod crossover;
pub mod builder;

pub use evolution::EvolutionConfig;
pub use mutation::MutationConfig;
pub use crossover::CrossoverConfig;
pub use builder::EvolutionBuilder;
```

### Configuration Hierarchy

```
EvolutionConfig (top-level)
├── Population settings
├── Speciation settings
├── MutationConfig
│   ├── Connection mutation rates
│   ├── Genome mutation rates
│   └── Parameter perturbation settings
└── CrossoverConfig
    ├── Compatibility coefficients
    └── Reproduction probabilities
```

---

## Implementation Steps

### Step 1: Create Config Structs

**File:** `src/config/evolution.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Minimum population size to maintain
    pub population_size: usize,
    
    /// Genetic distance threshold for speciation
    pub specie_threshold: f64,
    
    /// Generations without improvement before species truncation
    pub no_improvement_truncate: usize,
    
    /// Number of best individuals to preserve unchanged
    pub champion_preservation: usize,
    
    /// Ratio of offspring from mutation without crossover (0.0 to 1.0)
    pub reproduction_copy_ratio: f64,
    
    /// Nested mutation configuration
    pub mutation: MutationConfig,
    
    /// Nested crossover configuration
    pub crossover: CrossoverConfig,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 150,
            specie_threshold: 4.0,
            no_improvement_truncate: 10,
            champion_preservation: 1,
            reproduction_copy_ratio: 0.25,
            mutation: MutationConfig::default(),
            crossover: CrossoverConfig::default(),
        }
    }
}

impl EvolutionConfig {
    /// Validate configuration for internal consistency
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.population_size == 0 {
            return Err(ConfigError::InvalidPopulationSize);
        }
        if self.specie_threshold <= 0.0 {
            return Err(ConfigError::InvalidSpecieThreshold);
        }
        if self.reproduction_copy_ratio < 0.0 || self.reproduction_copy_ratio > 1.0 {
            return Err(ConfigError::InvalidRatio("reproduction_copy_ratio".into()));
        }
        
        self.mutation.validate()?;
        self.crossover.validate()?;
        Ok(())
    }
    
    /// Preset for classification tasks
    pub fn classification() -> Self {
        Self {
            specie_threshold: 3.0,
            no_improvement_truncate: 15,
            ..Default::default()
        }
    }
    
    /// Preset for reinforcement learning / control tasks
    pub fn control_tasks() -> Self {
        Self {
            specie_threshold: 4.5,
            reproduction_copy_ratio: 0.15,
            mutation: MutationConfig::aggressive(),
            ..Default::default()
        }
    }
}
```

**File:** `src/config/mutation.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationConfig {
    /// Probability of disabling a connection (0.0 to 1.0)
    pub connection_disable_prob: f64,
    
    /// Probability of mutating connection parameters (0.0 to 1.0)
    pub connection_mutate_param_prob: f64,
    
    /// Probability of replacing parameter vs perturbing (0.0 to 1.0)
    pub param_replace_prob: f64,
    
    /// Factor for parameter perturbation (typically 0.01 to 0.1)
    pub param_perturb_factor: f64,
    
    /// Probability of mutating individual connections in genome (0.0 to 1.0)
    pub genome_connection_mutate_prob: f64,
    
    /// Probability of adding new connection (0.0 to 1.0)
    pub new_connection_prob: f64,
    
    /// Probability of bisecting connection (adding node) (0.0 to 1.0)
    pub bisect_connection_prob: f64,
    
    /// Probability of mutating existing connections (0.0 to 1.0)
    pub mutate_connection_prob: f64,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            connection_disable_prob: 0.01,
            connection_mutate_param_prob: 0.99,
            param_replace_prob: 0.10,
            param_perturb_factor: 0.05,
            genome_connection_mutate_prob: 0.20,
            new_connection_prob: 0.05,
            bisect_connection_prob: 0.15,
            mutate_connection_prob: 0.80,
        }
    }
}

impl MutationConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let probs = [
            ("connection_disable_prob", self.connection_disable_prob),
            ("connection_mutate_param_prob", self.connection_mutate_param_prob),
            ("param_replace_prob", self.param_replace_prob),
            ("new_connection_prob", self.new_connection_prob),
            ("bisect_connection_prob", self.bisect_connection_prob),
            ("mutate_connection_prob", self.mutate_connection_prob),
        ];
        
        for (name, prob) in probs {
            if prob < 0.0 || prob > 1.0 {
                return Err(ConfigError::InvalidProbability(name.into(), prob));
            }
        }
        
        Ok(())
    }
    
    /// Aggressive mutation for exploration
    pub fn aggressive() -> Self {
        Self {
            new_connection_prob: 0.10,
            bisect_connection_prob: 0.20,
            param_perturb_factor: 0.10,
            ..Default::default()
        }
    }
    
    /// Conservative mutation for exploitation
    pub fn conservative() -> Self {
        Self {
            new_connection_prob: 0.02,
            bisect_connection_prob: 0.08,
            param_perturb_factor: 0.02,
            ..Default::default()
        }
    }
}
```

**File:** `src/config/crossover.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossoverConfig {
    /// Coefficient for excess genes in compatibility distance
    pub excess_coefficient: f64,
    
    /// Coefficient for disjoint genes in compatibility distance
    pub disjoint_coefficient: f64,
    
    /// Coefficient for parameter differences in compatibility distance
    pub param_coefficient: f64,
    
    /// Probability of picking gene from less fit parent (0.0 to 1.0)
    pub probability_pick_less_fit: f64,
    
    /// Probability of keeping disabled genes (0.0 to 1.0)
    pub probability_keep_disabled: f64,
    
    /// Threshold for genome size normalization
    pub normalization_threshold: f64,
}

impl Default for CrossoverConfig {
    fn default() -> Self {
        Self {
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            param_coefficient: 0.4,
            probability_pick_less_fit: 0.50,
            probability_keep_disabled: 0.75,
            normalization_threshold: 20.0,
        }
    }
}

impl CrossoverConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.normalization_threshold <= 0.0 {
            return Err(ConfigError::InvalidThreshold);
        }
        
        let probs = [
            ("probability_pick_less_fit", self.probability_pick_less_fit),
            ("probability_keep_disabled", self.probability_keep_disabled),
        ];
        
        for (name, prob) in probs {
            if prob < 0.0 || prob > 1.0 {
                return Err(ConfigError::InvalidProbability(name.into(), prob));
            }
        }
        
        Ok(())
    }
}
```

### Step 2: Implement Builder Pattern

**File:** `src/config/builder.rs`

```rust
use super::{EvolutionConfig, MutationConfig, CrossoverConfig};
use crate::{Connection, Genome, Scenario, Hook};
use rand::RngCore;

pub struct EvolutionBuilder<C: Connection, G: Genome<C>> {
    config: EvolutionConfig,
    hooks: Vec<Hook<C, G>>,
}

impl<C: Connection, G: Genome<C>> EvolutionBuilder<C, G> {
    pub fn new() -> Self {
        Self {
            config: EvolutionConfig::default(),
            hooks: Vec::new(),
        }
    }
    
    pub fn with_config(mut self, config: EvolutionConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.config.population_size = size;
        self
    }
    
    pub fn specie_threshold(mut self, threshold: f64) -> Self {
        self.config.specie_threshold = threshold;
        self
    }
    
    pub fn mutation_config(mut self, config: MutationConfig) -> Self {
        self.config.mutation = config;
        self
    }
    
    pub fn crossover_config(mut self, config: CrossoverConfig) -> Self {
        self.config.crossover = config;
        self
    }
    
    pub fn add_hook(mut self, hook: Hook<C, G>) -> Self {
        self.hooks.push(hook);
        self
    }
    
    pub fn evolve<S, I, A>(
        self,
        scenario: S,
        init: I,
        activation: A,
        rng: impl RngCore,
    ) -> Result<(Vec<Specie<C, G>>, usize), EvolutionError>
    where
        S: Scenario<C, G, A>,
        I: FnOnce((usize, usize)) -> (Vec<Specie<C, G>>, usize),
        A: Fn(f64) -> f64,
    {
        // Validate configuration
        self.config.validate()?;
        
        // Call evolve_with_config (new function to be implemented)
        crate::scenario::evolve_with_config(
            scenario,
            init,
            activation,
            rng,
            EvolutionHooks::new(self.hooks),
            self.config,
        )
    }
}
```

### Step 3: Refactor Core Functions

**Changes to `src/scenario.rs`:**

```rust
// Add new function that accepts config
pub fn evolve_with_config<...>(
    scenario: S,
    init: I,
    σ: A,
    mut rng: impl RngCore,
    hooks: EvolutionHooks<C, G>,
    config: EvolutionConfig,
) -> Result<(Vec<Specie<C, G>>, usize), EvolutionError> {
    // Use config.specie_threshold instead of SPECIE_THRESHOLD
    // Use config.no_improvement_truncate instead of NO_IMPROVEMENT_TRUNCATE
    // Pass config to population_reproduce
    // ... implementation
}

// Keep old evolve() for backward compatibility, call evolve_with_config
pub fn evolve<...>(
    scenario: S,
    init: I,
    σ: A,
    rng: impl RngCore,
    hooks: EvolutionHooks<C, G>,
) -> (Vec<Specie<C, G>>, usize) {
    evolve_with_config(
        scenario,
        init,
        σ,
        rng,
        hooks,
        EvolutionConfig::default(),
    ).unwrap() // For backward compatibility
}
```

**Changes to `src/population.rs`:**

```rust
// Update speciate to accept threshold
pub fn speciate<C: Connection, G: Genome<C>>(
    genomes: impl Iterator<Item = (G, f64)>,
    reprs: impl Iterator<Item = SpecieRepr<C>>,
    specie_threshold: f64,
) -> Vec<Specie<C, G>> {
    // Use specie_threshold parameter instead of const SPECIE_THRESHOLD
    // ... implementation
}
```

**Changes to `src/reproduce.rs`:**

```rust
pub fn population_reproduce<C: Connection, G: Genome<C>>(
    species: &[(Specie<C, G>, f64)],
    population: usize,
    inno_head: usize,
    rng: &mut impl RngCore,
    config: &EvolutionConfig,
) -> (Vec<G>, usize) {
    // Use config.reproduction_copy_ratio
    // Use config.champion_preservation
    // Pass mutation config to genome.mutate()
    // ... implementation
}
```

### Step 4: Update Traits to Accept Config

**Changes to `src/genome/mod.rs`:**

```rust
pub trait Genome<C: Connection>: ... {
    // Add new method with config
    fn mutate_with_config(
        &mut self, 
        rng: &mut impl RngCore, 
        innogen: &mut InnoGen,
        config: &MutationConfig,
    );
    
    // Keep old method for backward compatibility
    fn mutate(&mut self, rng: &mut impl RngCore, innogen: &mut InnoGen) {
        self.mutate_with_config(rng, innogen, &MutationConfig::default())
    }
}

pub trait Connection: ... {
    // Add new methods with config
    fn mutate_with_config(&mut self, rng: &mut impl RngCore, config: &MutationConfig);
    
    // Keep old method for backward compatibility
    fn mutate(&mut self, rng: &mut impl RngCore) {
        self.mutate_with_config(rng, &MutationConfig::default())
    }
}
```

### Step 5: Add Error Types

**File:** `src/config/error.rs`

```rust
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum ConfigError {
    InvalidPopulationSize,
    InvalidSpecieThreshold,
    InvalidRatio(String),
    InvalidProbability(String, f64),
    InvalidThreshold,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConfigError::InvalidPopulationSize => {
                write!(f, "Population size must be greater than 0")
            }
            ConfigError::InvalidSpecieThreshold => {
                write!(f, "Specie threshold must be greater than 0.0")
            }
            ConfigError::InvalidRatio(name) => {
                write!(f, "{} must be between 0.0 and 1.0", name)
            }
            ConfigError::InvalidProbability(name, val) => {
                write!(f, "{} must be between 0.0 and 1.0, got {}", name, val)
            }
            ConfigError::InvalidThreshold => {
                write!(f, "Threshold must be greater than 0.0")
            }
        }
    }
}

impl Error for ConfigError {}
```

---

## Migration Guide for Existing Users

### Before (Current API)
```rust
evolve(
    Xor {},
    |(i, o)| population_init::<C, G>(i, o, POPULATION),
    relu,
    default_rng(),
    EvolutionHooks::new(vec![Box::new(hook)]),
);
```

### After (New API)
```rust
// Option 1: Use defaults
EvolutionBuilder::new()
    .population_size(1000)
    .add_hook(Box::new(hook))
    .evolve(Xor {}, |(i, o)| population_init::<C, G>(i, o, 1000), relu, default_rng())
    .expect("Evolution failed");

// Option 2: Customize configuration
EvolutionBuilder::new()
    .with_config(EvolutionConfig::control_tasks())
    .specie_threshold(5.0)
    .mutation_config(MutationConfig::aggressive())
    .add_hook(Box::new(hook))
    .evolve(Xor {}, init_fn, relu, default_rng())
    .expect("Evolution failed");

// Option 3: Old API still works (backward compatible)
evolve(Xor {}, init_fn, relu, default_rng(), hooks);
```

---

## Testing Strategy

### Unit Tests
- Test each config struct's validation
- Test default configurations
- Test preset configurations
- Test builder pattern methods

### Integration Tests
- Test evolution with custom configs
- Test backward compatibility
- Test config serialization/deserialization

### Example Updates
- Update all examples to show new API
- Keep old API examples for comparison
- Add configuration showcase example

---

## Documentation Requirements

1. **Rustdoc for all config types**
   - Explain each parameter's effect
   - Provide value ranges and recommendations
   - Include usage examples

2. **Migration guide**
   - Step-by-step conversion instructions
   - Side-by-side comparisons
   - Common pitfalls

3. **Configuration tuning guide**
   - Parameter interactions
   - Domain-specific recommendations
   - Performance implications

---

## Timeline

- **Week 1:** Implement config structs and builder
- **Week 2:** Refactor core functions to use config
- **Week 3:** Update examples, tests, and documentation

---

## Success Criteria

- ✅ All hardcoded constants moved to config
- ✅ Builder API provides fluent configuration
- ✅ Old API remains functional (backward compatible)
- ✅ All tests pass
- ✅ Examples updated and documented
- ✅ Config validation prevents invalid states

---

## Future Enhancements

1. **Config file loading**: Load from TOML/YAML
2. **Config templates**: Share configurations easily
3. **Runtime tuning**: Adjust config mid-evolution
4. **Auto-tuning**: Grid search for best config
5. **Config visualization**: Show parameter effects

---

This implementation guide provides a concrete, step-by-step plan to add the most critical quality of life feature: runtime configuration. Once complete, it unlocks all other improvements by making the library truly parameterizable.
