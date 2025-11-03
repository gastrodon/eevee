# Hooks System Design
## Functional, Composable, Extensible

**Status:** Design Phase  
**Design Decision:** KEEP CURRENT FUNCTIONAL API  
**Last Updated:** 2025-11-01

---

## Current API (Perfect - No Changes)

The current functional hook system is well-designed and should be **preserved as-is**:

```rust
pub type Hook<C, G> = Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>>;

pub struct EvolutionHooks<C: Connection, G: Genome<C>> {
    hooks: Vec<Hook<C, G>>,
}

impl<C: Connection, G: Genome<C>> EvolutionHooks<C, G> {
    pub fn new(hooks: Vec<Hook<C, G>>) -> Self {
        Self { hooks }
    }
}
```

### Why This Design is Good

1. **Functional:** Hooks are just functions - composable and testable
2. **Flexible:** Users can define custom logic as closures
3. **Type-Safe:** `ControlFlow<()>` pattern is clear and extensible
4. **Runtime Configurable:** Hooks can be constructed dynamically
5. **No Boilerplate:** No trait implementations required

---

## Enhancement: Standard Hook Library

Add **convenience functions** that return hooks, without changing the core API:

### Design Principle

> Factory functions that return `Hook<C, G>` - not a new hook system

### Example Implementation

```rust
// src/hooks/mod.rs

/// Factory function that returns a hook
pub fn print_progress<C: Connection, G: Genome<C>>(
    interval: usize
) -> Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>> {
    Box::new(move |stats| {
        if stats.generation % interval == 0 {
            if let Some((_, fitness)) = stats.fittest() {
                println!("Generation {}: Best fitness = {:.4}, Species = {}", 
                    stats.generation, fitness, stats.species.len());
            }
        }
        ControlFlow::Continue(())
    })
}

/// Stops evolution when fitness threshold is reached
pub fn fitness_threshold<C: Connection, G: Genome<C>>(
    target: f64
) -> Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>> {
    Box::new(move |stats| {
        if stats.any_fitter_than(target) {
            println!("Target fitness {} reached at generation {}", target, stats.generation);
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
}

/// Stops evolution after max generations
pub fn generation_limit<C: Connection, G: Genome<C>>(
    max: usize
) -> Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>> {
    Box::new(move |stats| {
        if stats.generation >= max {
            println!("Generation limit {} reached", max);
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
}
```

### Usage

```rust
use eevee::hooks;

// Combine factory functions with custom hooks
let hooks = EvolutionHooks::new(vec![
    hooks::print_progress(10),
    hooks::fitness_threshold(500.0),
    hooks::generation_limit(1000),
    Box::new(|stats| {
        // Custom logic
        if stats.generation == 100 {
            println!("Checkpoint at generation 100");
        }
        ControlFlow::Continue(())
    }),
]);

evolve(scenario, init, activation, rng, hooks);
```

---

## Hook Combinators (Optional Enhancement)

Functional composition of hooks:

```rust
pub fn any_of<C: Connection, G: Genome<C>>(
    hooks: Vec<Hook<C, G>>
) -> Hook<C, G> {
    Box::new(move |stats| {
        for hook in &hooks {
            if hook(stats).is_break() {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    })
}

pub fn when<C: Connection, G: Genome<C>>(
    condition: impl Fn(&Stats<C, G>) -> bool + 'static,
    hook: Hook<C, G>
) -> Hook<C, G> {
    Box::new(move |stats| {
        if condition(stats) {
            hook(stats)
        } else {
            ControlFlow::Continue(())
        }
    })
}
```

---

## Standard Hook Library

### Monitoring Hooks

- `print_progress(interval)` - Print fitness every N generations
- `print_detailed(interval)` - Print full statistics
- `log_to_file(path, interval)` - Write to log file

### Stopping Conditions

- `fitness_threshold(target)` - Stop when fitness reached
- `generation_limit(max)` - Stop after N generations
- `early_stopping(patience)` - Stop if no improvement
- `plateau_detection(window, threshold)` - Stop if fitness plateaus

### Checkpointing

- `save_checkpoint(dir, interval)` - Save population periodically
- `save_best(path)` - Save best genome each generation

### Statistics Collection

- `collect_stats()` - Returns hook + statistics object
- `track_diversity()` - Monitor population diversity

---

## No Changes Needed

The current hook system is **already perfect** for the functional approach. We just need to:

1. ✅ Keep `Hook<C, G>` type alias
2. ✅ Keep `EvolutionHooks` struct
3. ✅ Keep `ControlFlow<()>` pattern
4. ✅ Add convenience factory functions
5. ✅ Document the pattern

---

## Implementation Priority

**Phase 1.5** (Week 3):
- Create `src/hooks.rs` module
- Implement 5-10 common hook factories
- Document usage patterns
- Add examples

This is a **low-risk, high-value** addition that respects the existing design.
