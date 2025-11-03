# Phase 1 Implementation Roadmap
## Configuration Abstraction Layer

**Status:** Starting Implementation  
**Start Date:** 2025-11-01  
**Target Completion:** 3 weeks

---

## Core Objectives

### 1. ✅ Keep Functional Hook System
The current hook system is **perfect** - functional, composable, and extensible.

**Action:** NO CHANGES to hook system core
- Keep `Hook<C, G>` type alias
- Keep `EvolutionHooks` struct  
- Keep `ControlFlow<()>` pattern
- Add convenience factory functions only

### 2. 🎯 Make CONST Values Configurable
Replace all hardcoded trait constants with runtime-configurable abstraction layer.

**Action:** Create configuration structs and thread through codebase
- 14+ hardcoded constants → runtime config
- Backward compatible abstraction
- Zero performance regression

---

## Week-by-Week Plan

### Week 1: Configuration Foundation

#### Days 1-2: Create Config Structs
- [ ] Create `src/config/mod.rs` module structure
- [ ] Implement `EvolutionConfig` struct
- [ ] Implement `MutationConfig` struct  
- [ ] Implement `CrossoverConfig` struct
- [ ] Add `Default` implementations matching current constants
- [ ] Add `Serialize`/`Deserialize` derives
- [ ] Write validation methods
- [ ] Add unit tests for validation

**Deliverable:** Pure data config structs with defaults matching current behavior

#### Days 3-5: Abstraction Layer
- [ ] Add `evolve_with_config()` to `scenario.rs`
- [ ] Update `speciate()` to accept threshold parameter
- [ ] Update `delta()` to accept config parameter
- [ ] Add probability pre-conversion to config structs
- [ ] Inline critical path functions
- [ ] Keep old `evolve()` calling new with defaults
- [ ] Write integration tests

**Deliverable:** Top-level functions accept config, old API still works

### Week 2: Thread Config Through Codebase

#### Days 1-2: Trait Updates
- [ ] Add `mutate_with_config()` to `Genome` trait
- [ ] Add `mutate_with_config()` to `Connection` trait
- [ ] Add default implementations calling new methods with defaults
- [ ] Update `WConnection` implementation
- [ ] Update `BWConnection` implementation
- [ ] Update `Recurrent<C>` implementation

**Deliverable:** Traits support config, backward compatible

#### Days 3-4: Internal Functions
- [ ] Update `population_reproduce()` to accept config
- [ ] Update `reproduce()` to accept mutation config
- [ ] Update `reproduce_crossover()` to use config
- [ ] Update `reproduce_copy()` to use config
- [ ] Update `crossover()` to accept crossover config
- [ ] Update `crossover_eq()` and `crossover_ne()` helpers

**Deliverable:** Config threaded through entire call chain

#### Day 5: Testing & Validation
- [ ] Run all existing tests (should pass unchanged)
- [ ] Add tests for config-aware methods
- [ ] Benchmark performance vs hardcoded constants
- [ ] Verify default config produces identical results
- [ ] Fix any regressions

**Deliverable:** All tests passing, no performance regression

### Week 3: Builder API & Polish

#### Days 1-2: Builder Pattern
- [ ] Implement `EvolutionBuilder` struct
- [ ] Add chainable configuration methods
- [ ] Add hook management methods
- [ ] Implement `.evolve()` method with validation
- [ ] Add error handling and reporting
- [ ] Write builder tests

**Deliverable:** Fluent builder API for configuration

#### Days 3-4: Hook Library & Examples
- [ ] Create `src/hooks.rs` module
- [ ] Implement `print_progress(interval)` factory
- [ ] Implement `fitness_threshold(target)` factory
- [ ] Implement `generation_limit(max)` factory
- [ ] Implement `save_checkpoint(dir, interval)` factory
- [ ] Update existing examples to show new API
- [ ] Create `examples/00_configuration_basics.rs`
- [ ] Create `examples/01_using_hooks.rs`

**Deliverable:** Standard hook library + updated examples

#### Day 5: Documentation & Release
- [ ] Write rustdoc for all new APIs
- [ ] Create migration guide
- [ ] Update README with quick start
- [ ] Write CHANGELOG entry
- [ ] Final testing and validation
- [ ] Prepare for review

**Deliverable:** Complete documentation, ready for review

---

## Key Design Decisions

### ✅ Decision 1: Keep Functional Hooks
**Rationale:** Current design is excellent - functional, composable, runtime-configurable  
**Action:** Add convenience factories, no core changes

### ✅ Decision 2: Config Structs as Pure Data
**Rationale:** Separate what to configure from how to configure it  
**Action:** Simple structs with `Default`, thread through call chain

### ✅ Decision 3: Backward Compatibility
**Rationale:** Don't break existing users  
**Action:** Old methods call new ones with default config

### ✅ Decision 4: Performance First
**Rationale:** No runtime overhead vs constants  
**Action:** Inline everything, pre-convert probabilities, benchmark

---

## Success Criteria

- [x] Research complete and documented
- [ ] All 14+ CONST values configurable at runtime
- [ ] Zero performance regression with default config
- [ ] All existing tests pass without modification
- [ ] Backward compatibility 100% maintained
- [ ] Builder API provides ergonomic configuration
- [ ] Hook library provides 4+ standard hooks
- [ ] 2+ new examples demonstrating features
- [ ] Complete rustdoc documentation
- [ ] Migration guide written

---

## Risk Mitigation

### Risk: Performance Regression
**Mitigation:** Benchmark every change, inline hot paths, pre-convert probabilities

### Risk: Breaking Changes
**Mitigation:** Keep old API, add new methods, test extensively

### Risk: Complexity
**Mitigation:** Simple config structs, clear threading pattern, good docs

---

## After Phase 1

### Phase 2: Observability (Weeks 4-5)
- Statistics collection system
- More hook factories
- Checkpoint management
- Visualization helpers

### Phase 3: Documentation (Weeks 6-7)
- Comprehensive examples
- Cookbook patterns
- Troubleshooting guide
- API documentation polish

---

## Daily Standup Format

Update this doc daily with:
- ✅ What was completed
- 🚧 What's in progress
- 🔴 Blockers or concerns
- 📊 Performance metrics
- 🧪 Test results

---

## Notes & Learnings

_Track decisions, insights, and learnings as implementation progresses_

### 2025-11-01
- ✅ Research phase complete
- ✅ Design documents created
- ✅ Clarified functional hook system should be preserved
- ✅ Focused on CONST → config abstraction layer
- 📝 Starting Week 1: Configuration Foundation
