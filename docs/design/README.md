# Design Documentation Index
## Phase 1: Configuration System Implementation

**Status:** Design Complete, Ready for Implementation  
**Last Updated:** 2025-11-01

---

## Quick Links

### Core Design Documents

1. **[context_config_system.md](context_config_system.md)** ⭐ **START HERE**
   - Context-based configuration with RwLock pattern
   - Environment variable specification (`EEVEE_*` prefix)
   - Complete catalog of 25+ configurable constants
   - Standard interface design
   - **This is the primary implementation spec**

2. **[hooks_design.md](hooks_design.md)**
   - Functional hook system (preserved as-is)
   - Convenience factory functions
   - Hook composition patterns

3. **[configuration_abstraction.md](configuration_abstraction.md)**
   - How to thread config through codebase
   - Migration strategy for removing trait constants
   - Performance optimization techniques

4. **[configuration_system.md](configuration_system.md)**
   - Overall architecture
   - Builder pattern design
   - Preset configurations

5. **[ROADMAP.md](ROADMAP.md)**
   - 3-week implementation timeline
   - Day-by-day tasks
   - Success criteria

---

## Design Philosophy

### 1. Context-Based Configuration

**Pattern:** Like Go's context, but for configuration
```rust
EeveeContext (Arc<RwLock<EeveeConfig>>)
    ↓ Pass through function calls
    ↓ Multi-reader, single-writer
    ↓ Hot-reloadable
```

### 2. Environment Variable Configuration

**Source of Truth:** Environment variables with `EEVEE_*` prefix
```bash
EEVEE_SPECIE_THRESHOLD=4.0
EEVEE_PARAM_PERTURB_FACTOR=0.05
# ... 25+ parameters
```

### 3. Functional Hook System (Preserved)

**Decision:** Keep current design - it's already perfect
```rust
Hook<C, G> = Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>>
```
Just add convenience factory functions.

---

## Configuration Catalog

### All Configurable Constants (25+)

| Category | Count | Examples |
|----------|-------|----------|
| Evolution | 1 | `NO_IMPROVEMENT_TRUNCATE` |
| Population | 1 | `SPECIE_THRESHOLD` |
| Connection Mutation | 8 | `PARAM_PERTURB_FAC`, `PROBABILITY_PICK_RL` |
| Genome Mutation | 5 | `MUTATE_CONNECTION_PROBABILITY` |
| Crossover | 4 | `EXCESS_COEFFICIENT`, `NORMALIZATION_THRESHOLD` |
| Reproduction | 2 | Copy ratio, champion preservation |
| Parameter Ranges | 2 | Mutation min/max values |

**Complete list:** See `context_config_system.md`

---

## Implementation Plan

### Week 1: Configuration Foundation
- Create config module structure
- Implement `EeveeContext` with RwLock
- Environment variable loader
- Validation layer

### Week 2: Thread Config Through Codebase
- Add `*_with_context()` methods to traits
- Update internal functions
- Maintain backward compatibility

### Week 3: Builder API & Polish
- Implement `EvolutionBuilder`
- Standard hook library
- Examples and documentation

**Detailed plan:** See `ROADMAP.md`

---

## Key Design Decisions

### ✅ Decision 1: Keep Functional Hooks
**Rationale:** Current design is excellent  
**Document:** `hooks_design.md`

### ✅ Decision 2: Context-Based Config
**Rationale:** Thread-safe, hot-reloadable, pass-through pattern  
**Document:** `context_config_system.md`

### ✅ Decision 3: Environment Variables
**Rationale:** 12-factor app pattern, no config files needed  
**Document:** `context_config_system.md`

### ✅ Decision 4: Backward Compatible
**Rationale:** Don't break existing users  
**Document:** `configuration_abstraction.md`

---

## Usage Examples

### Example 1: Load from Environment

```rust
use eevee::config::EeveeContext;

// Load config from EEVEE_* environment variables
let ctx = EeveeContext::from_env()?;

// Pass to evolution
evolve_with_context(scenario, init, activation, rng, hooks, ctx)?;
```

### Example 2: Programmatic Configuration

```rust
let ctx = EeveeContext::default();

// Update specific parameters
ctx.update(|config| {
    config.population.specie_threshold = 5.0;
    config.genome_mutation.new_connection_probability = 0.10;
});

evolve_with_context(scenario, init, activation, rng, hooks, ctx)?;
```

### Example 3: Standard Hooks

```rust
use eevee::hooks;

let hooks = EvolutionHooks::new(vec![
    hooks::print_progress(10),
    hooks::fitness_threshold(500.0),
    hooks::generation_limit(1000),
    Box::new(|stats| {
        // Custom logic
        ControlFlow::Continue(())
    }),
]);
```

---

## API Overview

### Core Types

```rust
// Context (thread-safe config container)
pub struct EeveeContext { ... }
impl EeveeContext {
    pub fn from_env() -> Result<Self, ConfigError>;
    pub fn default() -> Self;
    pub fn read(&self) -> RwLockReadGuard<'_, EeveeConfig>;
    pub fn write(&self) -> RwLockWriteGuard<'_, EeveeConfig>;
}

// Configuration (pure data)
pub struct EeveeConfig {
    pub evolution: EvolutionParams,
    pub population: PopulationParams,
    pub connection_mutation: ConnectionMutationParams,
    pub genome_mutation: GenomeMutationParams,
    pub crossover: CrossoverParams,
    pub reproduction: ReproductionParams,
}

// Hook factories (convenience functions)
pub mod hooks {
    pub fn print_progress(interval: usize) -> Hook<C, G>;
    pub fn fitness_threshold(target: f64) -> Hook<C, G>;
    pub fn generation_limit(max: usize) -> Hook<C, G>;
}
```

---

## Migration Guide

### Current Code (Still Works)

```rust
evolve(
    scenario,
    init_fn,
    activation,
    rng,
    EvolutionHooks::new(vec![Box::new(my_hook)]),
);
```

### New Code (Context-Aware)

```rust
let ctx = EeveeContext::from_env()?;
evolve_with_context(
    scenario,
    init_fn,
    activation,
    rng,
    EvolutionHooks::new(vec![
        hooks::print_progress(10),
        Box::new(my_hook),
    ]),
    ctx,
)?;
```

---

## Performance Considerations

### Zero-Cost Abstraction

1. **Pre-convert probabilities:** f64 → u64 at config creation
2. **Inline everything:** `#[inline(always)]` on hot paths
3. **Lock-free reads:** Consider `arc_swap` for read-heavy workloads
4. **Benchmark:** Verify no regression vs hardcoded constants

**Details:** See `configuration_abstraction.md`

---

## Testing Strategy

### Unit Tests
- Config validation
- Environment variable parsing
- Thread safety (multiple readers)

### Integration Tests
- Evolution with custom config
- Backward compatibility
- Config hot-reload

### Performance Tests
- Benchmark vs hardcoded constants
- Lock contention under load

**Details:** See `context_config_system.md`

---

## Success Criteria

- [ ] All 25+ constants configurable via environment variables
- [ ] Zero performance regression with default config
- [ ] 100% backward compatibility maintained
- [ ] Thread-safe multi-reader access
- [ ] Hot-reload capability
- [ ] Complete documentation and examples
- [ ] All tests passing

---

## Next Steps

1. Review design documents
2. Gather feedback on approach
3. Start Week 1 implementation (config module)
4. Update ROADMAP.md with progress

---

## Questions & Feedback

For design questions or suggestions, see the specific design document and refer to the rationale sections. All design decisions are documented with justification.

**Key Contacts:**
- Design Discussion: PR comments
- Implementation Questions: Check ROADMAP.md for current status
