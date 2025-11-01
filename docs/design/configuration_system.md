# Configuration System Design
## Phase 1: Foundation for Parameterizable Library

**Status:** Design Phase  
**Target:** v0.2.0  
**Last Updated:** 2025-11-01

---

## Goals

1. Enable runtime configuration of all algorithm parameters
2. Maintain backward compatibility with existing API
3. Provide ergonomic builder pattern for configuration
4. Support preset configurations for common use cases

## Non-Goals

1. Breaking changes to existing API
2. Performance regression vs hardcoded constants
3. Complex configuration file loading (future enhancement)

---

## Architecture Overview

### Configuration Hierarchy

```
EvolutionConfig
├── population_size: usize
├── specie_threshold: f64
├── no_improvement_truncate: usize
├── champion_preservation: usize
├── reproduction_copy_ratio: f64
├── MutationConfig
│   ├── connection_disable_prob: f64
│   ├── connection_mutate_param_prob: f64
│   ├── param_replace_prob: f64
│   ├── param_perturb_factor: f64
│   ├── genome_connection_mutate_prob: f64
│   ├── new_connection_prob: f64
│   ├── bisect_connection_prob: f64
│   └── mutate_connection_prob: f64
└── CrossoverConfig
    ├── excess_coefficient: f64
    ├── disjoint_coefficient: f64
    ├── param_coefficient: f64
    ├── probability_pick_less_fit: f64
    ├── probability_keep_disabled: f64
    └── normalization_threshold: f64
```

---

## API Design

### Current API (Preserved)

```rust
// This continues to work unchanged
evolve(
    scenario,
    init_fn,
    activation,
    rng,
    EvolutionHooks::new(vec![Box::new(my_hook)]),
);
```

### New Builder API

```rust
use eevee::config::EvolutionBuilder;

// Simple usage with minimal configuration
EvolutionBuilder::new()
    .population_size(1000)
    .specie_threshold(5.0)
    .add_hook(Box::new(my_hook))
    .evolve(scenario, init_fn, activation, rng)?;

// Advanced usage with full configuration
EvolutionBuilder::new()
    .with_config(EvolutionConfig {
        population_size: 1000,
        specie_threshold: 4.5,
        no_improvement_truncate: 15,
        champion_preservation: 2,
        reproduction_copy_ratio: 0.20,
        mutation: MutationConfig::aggressive(),
        crossover: CrossoverConfig::default(),
    })
    .add_hook(Box::new(progress_hook))
    .add_hook(Box::new(checkpoint_hook))
    .evolve(scenario, init_fn, activation, rng)?;

// Using presets
EvolutionBuilder::new()
    .with_config(EvolutionConfig::for_classification())
    .mutation_config(MutationConfig::aggressive())
    .specie_threshold(3.5)  // Override specific preset values
    .evolve(scenario, init_fn, activation, rng)?;
```

---

## Hook System Design

### Current Hook API (Preserved)

The current functional approach is maintained:

```rust
pub type Hook<C, G> = Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>>;

// Users can define hooks as closures
let my_hook = Box::new(|stats: &mut Stats<C, G>| {
    if stats.generation % 10 == 0 {
        println!("Generation {}: fitness {}", stats.generation, stats.fittest().unwrap().1);
    }
    if stats.any_fitter_than(500.0) {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    }
});
```

### Standard Hook Library (New)

Convenience functions that **return hooks** without changing the API:

```rust
use eevee::hooks;

// Factory functions that return Box<dyn Fn(...) -> ControlFlow<()>>
EvolutionBuilder::new()
    .add_hook(hooks::print_progress(10))              // Returns Hook<C, G>
    .add_hook(hooks::fitness_threshold(500.0))         // Returns Hook<C, G>
    .add_hook(hooks::generation_limit(1000))           // Returns Hook<C, G>
    .add_hook(hooks::save_checkpoint("checkpoints/", 50)) // Returns Hook<C, G>
    .add_hook(Box::new(|stats| {                       // Custom hook still works
        // Custom logic here
        ControlFlow::Continue(())
    }))
    .evolve(scenario, init_fn, activation, rng)?;
```

### Hook Composition

Hooks can be composed functionally:

```rust
use eevee::hooks;

// Combine multiple stopping conditions
let stop_hook = hooks::any_of(vec![
    hooks::fitness_threshold(500.0),
    hooks::generation_limit(1000),
    hooks::early_stopping(patience: 20),
]);

// Conditional hooks
let debug_hook = hooks::when(
    |stats| stats.generation > 100,
    hooks::print_detailed_stats()
);
```

---

## Implementation Phases

### Phase 1.1: Configuration Structs (Week 1)
- [ ] Create `src/config/mod.rs` module
- [ ] Implement `EvolutionConfig` with `Default`
- [ ] Implement `MutationConfig` with presets
- [ ] Implement `CrossoverConfig`
- [ ] Add validation methods
- [ ] Add `Serialize`/`Deserialize` support
- [ ] Write unit tests for validation

### Phase 1.2: Builder Pattern (Week 1-2)
- [ ] Implement `EvolutionBuilder`
- [ ] Add chainable configuration methods
- [ ] Add hook management
- [ ] Implement `.evolve()` method
- [ ] Write integration tests

### Phase 1.3: Core Refactoring (Week 2)
- [ ] Add `evolve_with_config()` to `scenario.rs`
- [ ] Update `speciate()` to accept threshold parameter
- [ ] Update `population_reproduce()` to accept config
- [ ] Thread configuration through call chain
- [ ] Maintain old `evolve()` for backward compatibility

### Phase 1.4: Trait Updates (Week 2-3)
- [ ] Add `*_with_config()` methods to `Genome` trait
- [ ] Add `*_with_config()` methods to `Connection` trait
- [ ] Implement config-aware mutation
- [ ] Implement config-aware crossover
- [ ] Keep old methods as default implementations

### Phase 1.5: Standard Hooks (Week 3)
- [ ] Create `src/hooks/mod.rs` module
- [ ] Implement `print_progress(interval)` hook factory
- [ ] Implement `fitness_threshold(target)` hook factory
- [ ] Implement `generation_limit(max)` hook factory
- [ ] Implement `save_checkpoint(path, interval)` hook factory
- [ ] Implement hook combinators (`any_of`, `all_of`, `when`)
- [ ] Write tests for hook factories

### Phase 1.6: Examples & Documentation (Week 3)
- [ ] Update existing examples to show new API
- [ ] Create `examples/00_configuration_basics.rs`
- [ ] Create `examples/01_using_presets.rs`
- [ ] Create `examples/02_custom_hooks.rs`
- [ ] Write migration guide
- [ ] Document all new APIs with rustdoc
- [ ] Update README with quick start

---

## Backward Compatibility Strategy

### Existing Code Continues to Work

```rust
// This exact code will continue to work unchanged
use eevee::{evolve, EvolutionHooks, /* ... */};

evolve(
    MyScenario,
    |(i, o)| population_init::<C, G>(i, o, 1000),
    relu,
    default_rng(),
    EvolutionHooks::new(vec![Box::new(my_hook)]),
);
```

**Implementation:** The old `evolve()` function calls `evolve_with_config()` with `EvolutionConfig::default()`.

### Trait Defaults

```rust
// Old trait methods remain, calling new config-aware versions with defaults
pub trait Genome<C: Connection> {
    // New method
    fn mutate_with_config(&mut self, rng: &mut impl RngCore, innogen: &mut InnoGen, config: &MutationConfig);
    
    // Old method (default implementation)
    fn mutate(&mut self, rng: &mut impl RngCore, innogen: &mut InnoGen) {
        self.mutate_with_config(rng, innogen, &MutationConfig::default())
    }
}
```

---

## Configuration Validation

### Design Principles

1. **Fail Fast:** Validate configuration at build time (before evolution starts)
2. **Clear Messages:** Provide actionable error messages
3. **Suggestions:** Suggest corrections when possible
4. **Type Safety:** Use types to prevent invalid states when possible

### Validation Rules

```rust
impl EvolutionConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Population size must be positive
        if self.population_size == 0 {
            return Err(ConfigError::InvalidPopulationSize {
                value: 0,
                suggestion: "Population size must be at least 1. Typical values: 50-1000".into(),
            });
        }
        
        // Specie threshold must be positive
        if self.specie_threshold <= 0.0 {
            return Err(ConfigError::InvalidSpecieThreshold {
                value: self.specie_threshold,
                suggestion: "Specie threshold must be positive. Typical values: 2.0-6.0".into(),
            });
        }
        
        // Reproduction ratio must be in [0, 1]
        if self.reproduction_copy_ratio < 0.0 || self.reproduction_copy_ratio > 1.0 {
            return Err(ConfigError::InvalidRatio {
                name: "reproduction_copy_ratio".into(),
                value: self.reproduction_copy_ratio,
                valid_range: (0.0, 1.0),
            });
        }
        
        // Validate nested configs
        self.mutation.validate()?;
        self.crossover.validate()?;
        
        Ok(())
    }
}
```

---

## Preset Configurations

### Design Rationale

Presets provide:
1. **Quick Start:** New users can get started without parameter tuning
2. **Best Practices:** Encode domain knowledge and proven configurations
3. **Documentation:** Show reasonable parameter ranges
4. **Baselines:** Provide starting points for experimentation

### Preset Catalog

```rust
impl EvolutionConfig {
    /// Conservative configuration for classification tasks
    /// - Slower speciation (threshold: 3.0)
    /// - More patience (truncate: 15)
    /// - Balanced mutation rates
    pub fn for_classification() -> Self {
        Self {
            specie_threshold: 3.0,
            no_improvement_truncate: 15,
            mutation: MutationConfig::balanced(),
            ..Default::default()
        }
    }
    
    /// Aggressive configuration for reinforcement learning
    /// - Faster speciation (threshold: 4.5)
    /// - More crossover, less copying (ratio: 0.15)
    /// - Higher mutation rates
    pub fn for_control_tasks() -> Self {
        Self {
            specie_threshold: 4.5,
            reproduction_copy_ratio: 0.15,
            mutation: MutationConfig::aggressive(),
            ..Default::default()
        }
    }
    
    /// Configuration for time series prediction
    /// - Moderate speciation
    /// - Conservative mutation
    /// - Emphasis on parameter refinement
    pub fn for_time_series() -> Self {
        Self {
            specie_threshold: 3.5,
            mutation: MutationConfig::conservative(),
            ..Default::default()
        }
    }
}

impl MutationConfig {
    pub fn aggressive() -> Self { /* ... */ }
    pub fn balanced() -> Self { Self::default() }
    pub fn conservative() -> Self { /* ... */ }
}
```

---

## Performance Considerations

### Design Goal: Zero-Cost Abstraction

The configuration system should have **no runtime overhead** compared to hardcoded constants when using default configurations.

### Optimization Strategies

1. **Inline Everything:** Mark config access methods as `#[inline]`
2. **Const Where Possible:** Use `const` for default values
3. **Benchmark Critical Paths:** Measure performance impact on hot loops
4. **Feature Flag:** Optionally compile with hardcoded constants for maximum performance

```rust
// Performance-critical code
#[inline(always)]
fn should_mutate(&self, rng: &mut impl RngCore, config: &MutationConfig) -> bool {
    rng.next_u64() < config.connection_mutate_param_prob_u64()
}

// Pre-convert probabilities to u64 for comparison
impl MutationConfig {
    #[inline(always)]
    fn connection_mutate_param_prob_u64(&self) -> u64 {
        (self.connection_mutate_param_prob * (u64::MAX as f64)) as u64
    }
}
```

---

## Testing Strategy

### Unit Tests

- Configuration validation (valid/invalid inputs)
- Default values match current behavior
- Preset configurations are valid
- Builder methods chain correctly

### Integration Tests

- Evolution with custom config produces expected results
- Old API still works (backward compatibility)
- Config serialization round-trips correctly
- Error messages are clear and helpful

### Regression Tests

- Performance: No slowdown vs hardcoded constants
- Behavior: Same results with default config vs old code
- Examples: All examples compile and run

---

## Migration Path

### For Library Users

1. **No Action Required:** Existing code continues to work
2. **Optional Upgrade:** New builder API available when needed
3. **Gradual Migration:** Can mix old and new APIs

### For Library Maintainers

1. **Week 1:** Implement config structs and builder
2. **Week 2:** Refactor core to accept config parameters
3. **Week 3:** Update examples and documentation
4. **Week 4:** Gather feedback and iterate

---

## Open Questions

1. **Config File Format:** TOML, YAML, or JSON for serialization? (Future)
2. **Runtime Tuning:** Allow config changes between generations? (Future)
3. **Auto-Tuning:** Implement parameter search? (Future)
4. **Deprecation:** Mark old functions as deprecated or keep indefinitely?

---

## Success Criteria

- [ ] All 14+ hardcoded constants replaceable at runtime
- [ ] Zero performance regression with default config
- [ ] All existing tests pass without modification
- [ ] All examples updated to show new API
- [ ] Migration guide written and reviewed
- [ ] Community feedback incorporated
- [ ] Documentation complete and clear

---

## References

- Research findings: `docs/research/research_findings.md`
- Implementation guide: `docs/research/implementation_guide.md`
- NEAT paper: Stanley & Miikkulainen (2002)
- Similar libraries: neat-python, NEAT-Rust
