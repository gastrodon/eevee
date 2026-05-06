# Eevee

Generalized NEAT neuroevolution toolkit in Rust. The local-minima problem that previously blocked XOR has been resolved (see `.claude/memory/project_eevee_overview.md` for the 5 fixes that landed). See `.claude/memory/` for full project context.

## Memory

Read `.claude/memory/MEMORY.md` at the start of each session.

## Build & Test

```sh
cargo build
cargo test
cargo test --all-features

# Run the XOR example (main test case for local minima debugging)
cargo run --example xor --features approx

# Benchmarks (requires gnuplot)
cargo bench <bench>
./cmp-bench <bench> [branch:-]   # compare across branches
./profile <bench>                 # flamegraph profiling
```

## Key Facts

- Nightly Rust required (`generic_const_exprs`, etc.)
- Does not work on Windows (`/dev/urandom` seeding)
- `parallel` feature enables rayon-based parallel genome evaluation
- Genome and connection types are generic — use `WConnection` + `Recurrent` for most work
- `SPECIE_THRESHOLD` in `src/population.rs` controls speciation sensitivity (currently `4.0` on main)
- The local-minima problem (formerly the primary open issue) was resolved on `worktree-fix-local-minima` (merged at `7c0b15e`) via 5 fix commits in speciate/reproduce/evolve plus 4 mutation/CTRNN tunings
