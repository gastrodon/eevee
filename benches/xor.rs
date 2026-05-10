#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use core::ops::ControlFlow;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    genome::{connection::BWConnection, NonRecurrent, Recurrent, WConnection},
    network::{
        activate::steep_sigmoid, BinaryFeedForward, FeedForward, Network, Realtime, RealtimeUnbias,
        ToNetwork,
    },
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    Connection, EvolutionConfig, Genome, Scenario, Stats,
};
use rand::Rng;
use std::marker::PhantomData;

const POPULATION: usize = 100;
const XOR_GEN_LIMIT: usize = 500;
const XOR_TARGET: f64 = 9.5; // 10 samples × 1.0 max each

const XOR_PAIRS: [([f64; 2], f64); 4] = [
    ([0., 0.], 1.),
    ([1., 1.], 1.),
    ([1., 0.], -1.),
    ([0., 1.], -1.),
];

struct Xor<NN: Network, C: Connection, G: Genome<C>> {
    _phantom: PhantomData<(NN, C, G)>,
}

impl<NN: Network, C: Connection, G: Genome<C>> Xor<NN, C, G> {
    fn new() -> Self {
        Xor {
            _phantom: PhantomData,
        }
    }
}

impl<NN: Network, C: Connection, G: Genome<C> + ToNetwork<NN, C>, A: Fn(f64) -> f64>
    Scenario<C, G, A> for Xor<NN, C, G>
{
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = genome.network();
        let mut fit = 0.;
        let mut rng = rand::rng();
        for _ in 0..10 {
            let (input, want) = XOR_PAIRS[rng.random_range(0..4)];
            network.step(&input, σ);
            let v = network.output()[0].tanh();
            fit += 1.0 - 0.5 * (want - v).abs();
        }
        fit
    }
}

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
// triple. Invoke once for recurrent combos and once for forward combos to avoid
// invalid cross-topology combinations (the trait bounds catch them at compile
// time, so mixing is a compile error, not a runtime one).
//
// Why recursive rules instead of nested $(...)+?
//   macro_rules! can't nest two independent repetition groups in a single
//   $(...) — all metavariables inside a repetition must share the same count.
//   The fix is TT-munching: peel off one Connection, call a helper that peels
//   off one Genome, then call a leaf rule that iterates the Network list freely.

macro_rules! xor_evolve_bench {
    // Entry point
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        xor_evolve_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        xor_evolve_bench!(@foreach_g $group, $C, $Gs, $Ns);
        xor_evolve_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        xor_evolve_bench!(@bench $group, $C, $G, $Ns);
        xor_evolve_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            $group.bench_function(
                concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                |b| {
                    b.iter_batched(
                        || population_init::<$C, $G<$C>>(2, 1, POPULATION),
                        |(pop, inno_head)| {
                            evolve(
                                Xor::<$N, $C, $G<$C>>::new(),
                                move |_| (pop, inno_head),
                                steep_sigmoid,
                                default_rng(),
                                EvolutionHooks::new(vec![
                                    Box::new(xor_stop_hook::<$C, $G<$C>>),
                                ]),
                                EvolutionConfig::default(),
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
    // smol_bench: criterion minimum — just enough for a flamegraph profile run.
    // full bench: 200 samples — expect multiple hours across all permutations.
    #[cfg(feature = "smol_bench")]
    group.sample_size(10);
    #[cfg(not(feature = "smol_bench"))]
    group.sample_size(200);

    // Recurrent genomes × recurrent networks
    xor_evolve_bench!(
        group,
        connections: [WConnection, BWConnection],
        genomes:     [Recurrent],
        networks:    [Realtime, RealtimeUnbias],
    );

    // Forward genomes × forward networks
    xor_evolve_bench!(
        group,
        connections: [WConnection, BWConnection],
        genomes:     [NonRecurrent],
        networks:    [FeedForward, BinaryFeedForward],
    );

    group.finish();
}

criterion_group!(benches, bench_xor);
criterion_main!(benches);
