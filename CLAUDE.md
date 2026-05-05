# Eevee

Generalized NEAT neuroevolution toolkit in Rust. WIP — the algorithm runs but gets stuck in local minima. See `.claude/memory/` for full project context.

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
- The local minima problem is the primary open issue — all active branches are investigating it
