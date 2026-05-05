---
name: Eevee project overview
description: What eevee is, its goals, architecture, and the major local minima bug
type: project
---

# Eevee — Generalized NEAT Neuroevolution Toolkit (Rust, WIP)

A Rust library implementing NEAT (NeuroEvolution of Augmenting Topologies) with the goal of generalizing it beyond the original paper — over arbitrary genome structures and arbitrary NN architectures. Pre-LLM code, written by the user.

**Why:** Explore and extend NEAT. Nothing "really works very well" per the README — topology search can run but is slow, inefficient, and often fails completely.

**How to apply:** Approach all suggestions with awareness that this is research/exploration code, not production. The core loop and traits are stable but the algorithm behavior is the problem.

---

## Core Architecture

The main evolution loop lives in `src/scenario.rs:evolve()`. Each generation:
1. Evaluate all genomes → fitness scores
2. Speciate (group by `delta` similarity)
3. Truncate species with no improvement in 10+ generations
4. Proportionally allocate population slots by adjusted fitness
5. Reproduce (elitism + copy + crossover + mutation)

**Key types and files:**
- `src/genome/mod.rs` — `Genome<C>` and `Connection` traits; `InnoGen` (innovation ID tracker)
- `src/genome/recurrent.rs` — `Recurrent<C>`: the only concrete genome; allows recurrent connections
- `src/genome/connection.rs` — `WConnection` (weight only), `BWConnection` (weight + per-connection bias)
- `src/crossover.rs` — `delta()` (speciation distance), `crossover()` (gene alignment + inheritance)
- `src/population.rs` — `speciate()`, `population_init()`, `Specie`, `SPECIE_THRESHOLD = 4.0`
- `src/reproduce.rs` — `reproduce()`, `population_reproduce()`; 1 elite + 25% copy + 75% crossover
- `src/network/` — three NN impls: `Continuous` (CTRNN, stateful), `NonBias` (CTRNN, no bias), `Simple` (walks connections oldest→newest on flat state)
- `src/random.rs` — `WyRng` (fast RNG lifted from smol-rs/fastrand); `seed_urandom()`; `EventKind` trait for probability tables

**Genome mutation events (probabilities):** 5% NewConnection, 15% BisectConnection, 80% MutateConnection, 0% MutateNode

**Connection mutation:** 1% Disable, 99% MutateParam (perturb by 5% of [-3,3] range, or 10% chance full replace)

**Speciation delta formula** (current main):
```
(DISJOINT_COEFF * disjoint + EXCESS_COEFF * excess) / fac + PARAM_COEFF * avg_param_diff
```
where `fac = max(len_l, len_r)` if >= 20 else 1.0. `SPECIE_THRESHOLD = 4.0`.

**Genome initialization:** `Recurrent::new()` creates ALL valid connections between (sensory+bias) → action nodes upfront (saturated initialization, added in 0.2.0).

**Examples:** `xor.rs` (XOR via Simple NN + relu), `sentiment/` (sentiment analysis via Continuous NN), `nes-tetris/` (NES Tetris via Continuous NN)

---

## Major Bug: Stuck in Local Minima

The algorithm converges to a local optimum and never escapes. Documented in `XOR_INVESTIGATION.md` (on `feature/crossover-no-fac` branch):

- XOR example gets stuck at ~198/400 fitness indefinitely
- Root cause: genomes with 0 connections output 0.0; two of the four XOR test cases (expect 0) are trivially correct, scoring 198 without any network structure
- Mutation rate for new connections is only 5%, and random weights initially hurt fitness → no evolutionary pressure to escape
- The XOR target is actually XNOR (`[0,0]→1, [1,1]→1, [1,0]→0, [0,1]→0`)

**A secondary (now partially fixed) speciation bug** was also found: the `delta` normalization by genome size made large genomes always appear similar, collapsing everything to 1 species. This was investigated and partially fixed on the branches.

---

## Branch Investigation Summary

All 9 open branches are trying to understand or fix the local minima problem:

**Shared base (8 commits on most branches, commit `1ae0059`):**
- Removed delta normalization (fixed speciation collapse for large genomes)
- Lowered SPECIE_THRESHOLD 4.0 → 3.0
- Added diverse initialization (`population_init_diverse`)
- Added first-generation species cap
- Replaced top-N selection with weighted random reproduction
- Created many diagnostic examples: `xor_diagnostic`, `xor_delta_analysis`, `xor_genome_sizes`, `xor_debug`, `xor_test`

**Individual branches beyond the shared base:**
- `feature/specie-threshold-1p01`: lowers threshold further to 1.01 (aggressive speciation experiment)
- `feature/xor-diverse-refresh`: integrates diverse init into main xor.rs example
- `feature/weighted-reproduction-tests`: adds epsilon guard + deterministic tests for weighted reproduction; adds `seed_time`, `seed_pid_time`, `seed_thread_time` to random.rs
- `more-rng-seed`: same new seed functions, plus seed benchmarks
- `probabalistic-specie-survival`: changes species allocation from deterministic to probabilistic; young/struggling species get exponential-decay survival odds (`exp(-(age/5)^2)`) — directly attacking the local minima by keeping diverse species alive longer
- `bisect-intra-connection`: when bisecting `A→B` into `A→C→B`, also adds a random "interconnect" connection to/from C; adds `ConnectionPoint` enum for constrained path search in `open_path`
- `copilot/investigate-specie-convergence-issue`: just adds a seed benchmark scaffold
- `feature/crossover-no-fac` and `feature/examples-leftover`: identical, both are just the shared base tip
