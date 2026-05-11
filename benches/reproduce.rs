use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    genome::{InnoGen, Recurrent, WConnection},
    population::speciate,
    random::default_rng,
    reproduce::{population_reproduce, reproduce},
    Connection, Genome, SerializeFile,
};
use std::{fs, path::PathBuf};

fn load_fixture<C: Connection, G: Genome<C> + SerializeFile>(perm_id: &str) -> Vec<G> {
    let dir = PathBuf::from("benches/fixture").join(perm_id);
    assert!(
        dir.exists(),
        "fixture '{perm_id}' not found; run: cargo run --example xor_generate_fixture --features serialize_json"
    );
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read fixture dir: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| G::from_file(p).unwrap_or_else(|e| panic!("cannot parse {}: {e}", p.display())))
        .collect()
}

fn inno_head<C: Connection, G: Genome<C>>(genomes: &[G]) -> usize {
    genomes
        .iter()
        .flat_map(|g| g.connections().iter().map(|c| c.inno()))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// mutate: single genome mutation step (connection mutate / node insert / etc.)
// ---------------------------------------------------------------------------

macro_rules! mutate_bench {
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        mutate_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        mutate_bench!(@foreach_g $group, $C, $Gs, $Ns);
        mutate_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        mutate_bench!(@bench $group, $C, $G, $Ns);
        mutate_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            {
                let perm_id = concat!(stringify!($C), "_", stringify!($G), "_", stringify!($N));
                let genomes: Vec<$G<$C>> = load_fixture(perm_id);
                let head = inno_head::<$C, $G<$C>>(&genomes);

                $group.bench_function(
                    concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter_batched(
                            || (genomes[0].clone(), InnoGen::new(head), default_rng()),
                            |(mut g, mut inno, mut rng)| {
                                let _ = g.mutate(&mut rng, &mut inno);
                                (g, inno, rng)
                            },
                            BatchSize::SmallInput,
                        )
                    },
                );
            }
        )*
    };
}

// ---------------------------------------------------------------------------
// reproduce: elite + copy + crossover for a single specie.
// Fitness is assigned by index so the fittest genome is deterministic.
// ---------------------------------------------------------------------------

macro_rules! reproduce_bench {
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        reproduce_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        reproduce_bench!(@foreach_g $group, $C, $Gs, $Ns);
        reproduce_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        reproduce_bench!(@bench $group, $C, $G, $Ns);
        reproduce_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            {
                let perm_id = concat!(stringify!($C), "_", stringify!($G), "_", stringify!($N));
                let genomes: Vec<$G<$C>> = load_fixture(perm_id);
                let head = inno_head::<$C, $G<$C>>(&genomes);
                let n = genomes.len();

                $group.bench_function(
                    concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter_batched(
                            || {
                                let members: Vec<($G<$C>, f64)> = genomes
                                    .iter()
                                    .enumerate()
                                    .map(|(i, g)| (g.clone(), i as f64))
                                    .collect();
                                (members, InnoGen::new(head), default_rng())
                            },
                            |(members, mut inno, mut rng)| {
                                reproduce(members, n, &mut inno, &mut rng).unwrap()
                            },
                            BatchSize::SmallInput,
                        )
                    },
                );
            }
        )*
    };
}

// ---------------------------------------------------------------------------
// population_reproduce: full speciated-population reproduction pipeline.
// Species are constructed once outside the measurement.
// ---------------------------------------------------------------------------

macro_rules! population_reproduce_bench {
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        population_reproduce_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        population_reproduce_bench!(@foreach_g $group, $C, $Gs, $Ns);
        population_reproduce_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        population_reproduce_bench!(@bench $group, $C, $G, $Ns);
        population_reproduce_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            {
                let perm_id = concat!(stringify!($C), "_", stringify!($G), "_", stringify!($N));
                let genomes: Vec<$G<$C>> = load_fixture(perm_id);
                let head = inno_head::<$C, $G<$C>>(&genomes);
                let n = genomes.len();

                let species = speciate(
                    genomes.iter().enumerate().map(|(i, g)| (g.clone(), i as f64)),
                    std::iter::empty(),
                );

                $group.bench_function(
                    concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter_batched(
                            || default_rng(),
                            |mut rng| population_reproduce(&species, n, head, &mut rng),
                            BatchSize::SmallInput,
                        )
                    },
                );
            }
        )*
    };
}

fn bench_mutate(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutate");
    mutate_bench!(
        group,
        connections: [WConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );
    group.finish();
}

fn bench_reproduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("reproduce");
    reproduce_bench!(
        group,
        connections: [WConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );
    group.finish();
}

fn bench_population_reproduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("population_reproduce");
    population_reproduce_bench!(
        group,
        connections: [WConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_mutate,
    bench_reproduce,
    bench_population_reproduce
);
criterion_main!(benches);
