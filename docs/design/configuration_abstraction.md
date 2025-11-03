# Configuration Abstraction Layer
## Making CONST Values Runtime-Configurable

**Status:** Design Phase  
**Goal:** Replace hardcoded CONST with configurable parameters  
**Last Updated:** 2025-11-01

---

## Problem Statement

Currently, all algorithm parameters are hardcoded as trait constants:

```rust
// genome/mod.rs
pub trait Connection {
    const PROBABILITIES: [u64; 2] = [percent(1), percent(99)];
    const PARAM_REPLACE_PROBABILITY: u64 = percent(10);
    const PARAM_PERTURB_FAC: f64 = 0.05;
    const EXCESS_COEFFICIENT: f64 = 1.0;
    const DISJOINT_COEFFICIENT: f64 = 1.0;
    const PARAM_COEFFICIENT: f64 = 0.4;
    // ...
}

pub trait Genome<C: Connection> {
    const MUTATE_NODE_PROBABILITY: u64 = percent(20);
    const MUTATE_CONNECTION_PROBABILITY: u64 = percent(20);
    const PROBABILITIES: [u64; 4] = [percent(5), percent(15), percent(80), percent(0)];
}

// scenario.rs
const NO_IMPROVEMENT_TRUNCATE: usize = 10;

// population.rs
const SPECIE_THRESHOLD: f64 = 4.;
```

**Users cannot change these without modifying source code.**

---

## Design: Abstract Configuration Layer

### Core Principle

> Separate **what** to configure from **how** to configure it

### Architecture

```
User Code
    ↓
Configuration Structs (what)
    ↓
Abstraction Layer (bridge)
    ↓
Algorithm Implementation (how)
```

---

## Configuration Structs (Pure Data)

Simple, serializable configuration structs:

```rust
// src/config/mod.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub specie_threshold: f64,
    pub no_improvement_truncate: usize,
    pub champion_preservation: usize,
    pub reproduction_copy_ratio: f64,
    pub mutation: MutationConfig,
    pub crossover: CrossoverConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationConfig {
    // Connection-level mutation
    pub connection_disable_prob: f64,
    pub connection_mutate_param_prob: f64,
    pub param_replace_prob: f64,
    pub param_perturb_factor: f64,
    
    // Genome-level mutation
    pub genome_connection_mutate_prob: f64,
    pub new_connection_prob: f64,
    pub bisect_connection_prob: f64,
    pub mutate_connection_prob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossoverConfig {
    pub excess_coefficient: f64,
    pub disjoint_coefficient: f64,
    pub param_coefficient: f64,
    pub probability_pick_less_fit: f64,
    pub probability_keep_disabled: f64,
    pub normalization_threshold: f64,
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
```

---

## Abstraction Layer Design

### Strategy 1: Pass Config Through Call Chain

Thread configuration through the function call hierarchy:

```rust
// Current signature
pub fn evolve<C, G, I, A, S>(
    scenario: S,
    init: I,
    σ: A,
    rng: impl RngCore,
    hooks: EvolutionHooks<C, G>,
) -> (Vec<Specie<C, G>>, usize)

// New signature (config-aware)
pub fn evolve_with_config<C, G, I, A, S>(
    scenario: S,
    init: I,
    σ: A,
    rng: impl RngCore,
    hooks: EvolutionHooks<C, G>,
    config: EvolutionConfig,
) -> Result<(Vec<Specie<C, G>>, usize), ConfigError>

// Old function calls new one with defaults (backward compatible)
pub fn evolve<C, G, I, A, S>(
    scenario: S,
    init: I,
    σ: A,
    rng: impl RngCore,
    hooks: EvolutionHooks<C, G>,
) -> (Vec<Specie<C, G>>, usize) {
    evolve_with_config(scenario, init, σ, rng, hooks, EvolutionConfig::default())
        .expect("Evolution failed with default config")
}
```

### Strategy 2: Config-Aware Methods

Add config-aware versions of methods, keep old ones for compatibility:

```rust
pub trait Genome<C: Connection> {
    // New: config-aware
    fn mutate_with_config(
        &mut self,
        rng: &mut impl RngCore,
        innogen: &mut InnoGen,
        config: &MutationConfig,
    );
    
    // Old: backward compatible (default impl)
    fn mutate(&mut self, rng: &mut impl RngCore, innogen: &mut InnoGen) {
        self.mutate_with_config(rng, innogen, &MutationConfig::default())
    }
}

pub trait Connection {
    // New: config-aware
    fn mutate_with_config(&mut self, rng: &mut impl RngCore, config: &MutationConfig);
    
    // Old: backward compatible (default impl)
    fn mutate(&mut self, rng: &mut impl RngCore) {
        self.mutate_with_config(rng, &MutationConfig::default())
    }
}
```

### Strategy 3: Remove Trait Constants

Trait constants become function parameters:

```rust
// Before: Hardcoded
pub trait Connection {
    const EXCESS_COEFFICIENT: f64;
    const DISJOINT_COEFFICIENT: f64;
    const PARAM_COEFFICIENT: f64;
}

// After: Parameterized
pub fn delta<C: Connection>(
    l: &[C],
    r: &[C],
    config: &CrossoverConfig,
) -> f64 {
    // Use config.excess_coefficient instead of C::EXCESS_COEFFICIENT
    let (disjoint, excess) = disjoint_excess_count(l, r);
    (config.disjoint_coefficient * disjoint + config.excess_coefficient * excess) / fac
        + config.param_coefficient * avg_param_diff(l, r)
}
```

---

## Threading Configuration Through Codebase

### Level 1: Top-Level (scenario.rs)

```rust
pub fn evolve_with_config<...>(
    scenario: S,
    init: I,
    σ: A,
    mut rng: impl RngCore,
    hooks: EvolutionHooks<C, G>,
    config: EvolutionConfig,
) -> Result<(Vec<Specie<C, G>>, usize), ConfigError> {
    config.validate()?;
    
    // Use config.specie_threshold
    let species = speciate(genomes, reprs, config.specie_threshold);
    
    // Use config.no_improvement_truncate
    if gen_achieved + config.no_improvement_truncate <= gen_idx {
        // truncate species
    }
    
    // Pass config down
    (pop_flat, inno_head) = population_reproduce(
        &p_scored,
        population_lim,
        inno_head,
        &mut rng,
        &config,  // <-- Pass config
    );
    
    Ok((species, inno_head))
}
```

### Level 2: Population (reproduce.rs)

```rust
pub fn population_reproduce<C: Connection, G: Genome<C>>(
    species: &[(Specie<C, G>, f64)],
    population: usize,
    inno_head: usize,
    rng: &mut impl RngCore,
    config: &EvolutionConfig,  // <-- Accept config
) -> (Vec<G>, usize) {
    // Use config.reproduction_copy_ratio
    let size_copy = (size as f64 * config.reproduction_copy_ratio) as usize;
    
    // Use config.champion_preservation
    for i in 0..config.champion_preservation {
        pop.push(/* champion */);
    }
    
    // Pass config to reproduce
    reproduce(genomes, size, &mut innogen, rng, &config.mutation)
}

fn reproduce<C: Connection, G: Genome<C>>(
    genomes: Vec<(G, f64)>,
    size: usize,
    innogen: &mut InnoGen,
    rng: &mut impl RngCore,
    mutation_config: &MutationConfig,  // <-- Accept mutation config
) -> Result<Vec<G>, Box<dyn Error>> {
    // ...
    let mut child = l.reproduce_with(r, std::cmp::Ordering::Greater, rng);
    child.mutate_with_config(rng, innogen, mutation_config);  // <-- Use config
    // ...
}
```

### Level 3: Genome (genome/mod.rs)

```rust
impl<C: Connection> Genome<C> for Recurrent<C> {
    fn mutate_with_config(
        &mut self,
        rng: &mut impl RngCore,
        innogen: &mut InnoGen,
        config: &MutationConfig,
    ) {
        // Convert probabilities to u64 for comparison
        let new_conn_prob = (config.new_connection_prob * u64::MAX as f64) as u64;
        let bisect_prob = (config.bisect_connection_prob * u64::MAX as f64) as u64;
        let mutate_prob = (config.mutate_connection_prob * u64::MAX as f64) as u64;
        
        let roll = rng.next_u64();
        
        if roll < new_conn_prob {
            self.new_connection(rng, innogen);
        } else if roll < new_conn_prob + bisect_prob {
            if !self.connections().is_empty() {
                self.bisect_connection(rng, innogen);
            }
        } else if roll < new_conn_prob + bisect_prob + mutate_prob {
            if !self.connections().is_empty() {
                self.mutate_connection_with_config(rng, config);
            }
        }
    }
    
    fn mutate_connection_with_config(
        &mut self,
        rng: &mut impl RngCore,
        config: &MutationConfig,
    ) {
        let prob = (config.genome_connection_mutate_prob * u64::MAX as f64) as u64;
        for c in self.connections_mut() {
            if rng.next_u64() < prob {
                c.mutate_with_config(rng, config);
            }
        }
    }
}
```

### Level 4: Connection (genome/connection.rs)

```rust
impl Connection for WConnection {
    fn mutate_with_config(&mut self, rng: &mut impl RngCore, config: &MutationConfig) {
        let disable_prob = (config.connection_disable_prob * u64::MAX as f64) as u64;
        let mutate_prob = (config.connection_mutate_param_prob * u64::MAX as f64) as u64;
        
        let roll = rng.next_u64();
        
        if roll < disable_prob {
            self.disable();
        } else if roll < disable_prob + mutate_prob {
            self.mutate_param_with_config(rng, config);
        }
    }
    
    fn mutate_param_with_config(&mut self, rng: &mut impl RngCore, config: &MutationConfig) {
        let replace_prob = (config.param_replace_prob * u64::MAX as f64) as u64;
        let replace = rng.next_u64() < replace_prob;
        
        let v: f64 = rng.sample(Uniform::new_inclusive(-3., 3.).unwrap());
        
        if replace {
            self.weight = v;
        } else {
            self.weight += config.param_perturb_factor * v;
        }
    }
}
```

### Level 5: Crossover (crossover.rs)

```rust
pub fn delta<C: Connection>(
    l: &[C],
    r: &[C],
    config: &CrossoverConfig,  // <-- Accept config
) -> f64 {
    let l_size = l.len() as f64;
    let r_size = r.len() as f64;
    
    let fac = {
        let longest = f64::max(l_size, r_size);
        if longest < config.normalization_threshold {  // <-- Use config
            1.
        } else {
            longest
        }
    };

    if l_size == 0. || r_size == 0. {
        (config.excess_coefficient * f64::max(l_size, r_size)) / fac
    } else {
        let (disjoint, excess) = disjoint_excess_count(l, r);
        (config.disjoint_coefficient * disjoint + config.excess_coefficient * excess) / fac
            + config.param_coefficient * avg_param_diff(l, r)
    }
}

pub fn crossover<C: Connection>(
    l: &[C],
    r: &[C],
    l_fit: Ordering,
    rng: &mut impl RngCore,
    config: &CrossoverConfig,  // <-- Accept config
) -> Vec<C> {
    let mut usort = match l_fit {
        Ordering::Equal => crossover_eq(l, r, rng, config),
        Ordering::Less => crossover_ne(r, l, rng, config),
        Ordering::Greater => crossover_ne(l, r, rng, config),
    };
    usort.sort_by_key(|c| c.inno());
    usort
}
```

### Level 6: Population (population.rs)

```rust
pub fn speciate<C: Connection, G: Genome<C>>(
    genomes: impl Iterator<Item = (G, f64)>,
    reprs: impl Iterator<Item = SpecieRepr<C>>,
    specie_threshold: f64,  // <-- Accept threshold as parameter
) -> Vec<Specie<C, G>> {
    let mut sp = Vec::from_iter(reprs.map(|repr| Specie {
        repr,
        members: Vec::new(),
    }));

    for (genome, fitness) in genomes {
        match sp.iter_mut().find(|Specie { repr, .. }| {
            repr.delta(genome.connections()) < specie_threshold  // <-- Use parameter
        }) {
            Some(Specie { members, .. }) => members.push((genome, fitness)),
            None => {
                sp.push(Specie {
                    repr: SpecieRepr::new(genome.connections().to_vec()),
                    members: vec![(genome, fitness)],
                });
            }
        }
    }

    sp
}
```

---

## Migration Path

### Step 1: Add Config Structs (No Breaking Changes)

```rust
// src/config/mod.rs - NEW MODULE
pub struct EvolutionConfig { /* ... */ }
pub struct MutationConfig { /* ... */ }
pub struct CrossoverConfig { /* ... */ }
```

### Step 2: Add *_with_config Methods (Backward Compatible)

```rust
// genome/mod.rs
pub trait Genome<C: Connection> {
    fn mutate_with_config(&mut self, ...);  // NEW
    fn mutate(&mut self, ...) {             // OLD - calls new with defaults
        self.mutate_with_config(..., &MutationConfig::default())
    }
}
```

### Step 3: Add evolve_with_config (Backward Compatible)

```rust
// scenario.rs
pub fn evolve_with_config(..., config: EvolutionConfig) -> Result<...>;  // NEW
pub fn evolve(...) -> (...) {  // OLD - calls new with defaults
    evolve_with_config(..., EvolutionConfig::default()).unwrap()
}
```

### Step 4: Update Internal Functions

```rust
// Internal functions can be breaking (not public API)
fn population_reproduce(..., config: &EvolutionConfig) { /* ... */ }
pub fn speciate(..., threshold: f64) { /* ... */ }
fn delta(..., config: &CrossoverConfig) { /* ... */ }
```

---

## Performance Optimization

### Probability Pre-Conversion

Convert f64 probabilities to u64 once at config creation:

```rust
#[derive(Debug, Clone)]
pub struct MutationConfig {
    // Public f64 API
    pub connection_disable_prob: f64,
    pub connection_mutate_param_prob: f64,
    
    // Internal u64 cache
    connection_disable_prob_u64: u64,
    connection_mutate_param_prob_u64: u64,
}

impl MutationConfig {
    pub fn new(/* params */) -> Self {
        Self {
            connection_disable_prob,
            connection_mutate_param_prob,
            connection_disable_prob_u64: (connection_disable_prob * u64::MAX as f64) as u64,
            connection_mutate_param_prob_u64: (connection_mutate_param_prob * u64::MAX as f64) as u64,
        }
    }
    
    #[inline(always)]
    pub(crate) fn connection_disable_prob_u64(&self) -> u64 {
        self.connection_disable_prob_u64
    }
}
```

### Inline Everything

```rust
#[inline(always)]
fn should_mutate(&self, rng: &mut impl RngCore, config: &MutationConfig) -> bool {
    rng.next_u64() < config.connection_mutate_param_prob_u64()
}
```

---

## Summary: Configuration Abstraction Strategy

1. **Create Pure Config Structs** - Simple data containers
2. **Add *_with_config Methods** - Config-aware versions
3. **Keep Old Methods** - Backward compatible defaults
4. **Thread Config Through** - Pass down call chain
5. **Remove Trait Constants** - Replace with function parameters
6. **Optimize Hot Paths** - Inline, pre-convert probabilities

This approach makes **CONST values runtime-configurable** while maintaining backward compatibility and performance.
