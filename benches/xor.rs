#![allow(mixed_script_confusables)]

use core::ops::ControlFlow;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    genome::{connection::BWConnection, Recurrent, WConnection},
    network::{activate::steep_sigmoid, Continuous, NonBias},
    playground::xor::{XorScenario, XOR_TARGET},
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    Connection, Genome, Stats,
};

const POPULATION: usize = 100;
const XOR_GEN_LIMIT: usize = 500;

fn xor_stop_hook<C: Connection, G: Genome<C>>(stats: &mut Stats<'_, C, G>) -> ControlFlow<()> {
    if stats.any_fitter_than(XOR_TARGET) || stats.generation >= XOR_GEN_LIMIT {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// Matrix benchmark macro
// ---------------------------------------------------------------------------
//
// Generates one criterion benchmark per (Connection, Genome<Connection>, Network)
// triple by registering each with the supplied BenchmarkGroup expression.
//
// Usage:
//   xor_evolve_bench!(
//       group,
//       connections: [WConnection, BWConnection],
//       genomes:     [Recurrent],               // each becomes Recurrent<C>
//       networks:    [Continuous, NonBias],      // plain types — used as-is
//   );
//
// Why recursive rules instead of nested $(...)+?
//   macro_rules! can't nest two *independent* repetition groups in a single
//   $(...) — all metavariables inside a repetition must come from the same
//   capture group and share the same count.  The fix is TT-munching: peel off
//   one Connection, call a helper that peels off one Genome, then call a leaf
//   rule that can freely iterate the Network list because C and G are now
//   single idents (not in any repetition group).

macro_rules! xor_evolve_bench {
    // Entry point: normalise lists, hand off to @foreach_c
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        xor_evolve_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    // @foreach_c — base case: no connections left
    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};

    // @foreach_c — peel off the first C, recurse for the rest
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        xor_evolve_bench!(@foreach_g $group, $C, $Gs, $Ns);
        xor_evolve_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    // @foreach_g — base case: no genomes left for this C
    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};

    // @foreach_g — peel off the first G, recurse for the rest
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        xor_evolve_bench!(@bench $group, $C, $G, $Ns);
        xor_evolve_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    // @bench — C and G are now concrete; iterate the network list freely
    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            $group.bench_function(
                concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                |b| {
                    b.iter_batched(
                        || population_init::<$C, $G<$C>>(2, 1, POPULATION),
                        |(pop, inno_head)| {
                            evolve(
                                XorScenario::<$N>::default(),
                                move |_| (pop, inno_head),
                                steep_sigmoid,
                                default_rng(),
                                EvolutionHooks::new(vec![
                                    Box::new(xor_stop_hook::<$C, $G<$C>>),
                                ]),
                            )
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        )*
    };
}

fn bench_xor(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor");
    // Evolution runs can take seconds each; keep sample count low so the
    // suite finishes in a reasonable time.
    group.sample_size(10);

    xor_evolve_bench!(
        group,
        connections: [WConnection, BWConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );

    group.finish();
}

criterion_group!(benches, bench_xor);
criterion_main!(benches);
