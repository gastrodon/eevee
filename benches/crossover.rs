use core::cmp::Ordering;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    crossover::{avg_param_diff, crossover, delta, disjoint_excess_count},
    genome::{Recurrent, WConnection},
    random::default_rng,
    Connection, Genome, SerializeFile,
};
use std::{fs, path::PathBuf};

fn load_fixture<C: Connection, G: Genome<C> + SerializeFile>(perm_id: &str) -> Vec<G> {
    let dir = PathBuf::from("target/fixtures/xor").join(perm_id);
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

// ---------------------------------------------------------------------------
// crossover: merge two connection slices via NEAT crossover.
// Allocates a new Vec<C> each call — allocation cost is intentionally included.
// ---------------------------------------------------------------------------

macro_rules! crossover_bench {
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        crossover_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        crossover_bench!(@foreach_g $group, $C, $Gs, $Ns);
        crossover_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        crossover_bench!(@bench $group, $C, $G, $Ns);
        crossover_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            {
                let perm_id = concat!(stringify!($C), "_", stringify!($G), "_", stringify!($N));
                let genomes: Vec<$G<$C>> = load_fixture(perm_id);
                let l = genomes[0].connections().to_vec();
                let r = genomes[1 % genomes.len()].connections().to_vec();

                let mut rng = default_rng();
                $group.bench_function(
                    concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter_batched(
                            || (l.clone(), r.clone()),
                            |(l, r)| crossover(&l, &r, Ordering::Greater, &mut rng),
                            BatchSize::SmallInput,
                        )
                    },
                );
            }
        )*
    };
}

// ---------------------------------------------------------------------------
// disjoint_excess_count: sorted merge scan counting misaligned innovations.
// Pure read — no allocation.
// ---------------------------------------------------------------------------

macro_rules! alignment_bench {
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        alignment_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        alignment_bench!(@foreach_g $group, $C, $Gs, $Ns);
        alignment_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        alignment_bench!(@bench $group, $C, $G, $Ns);
        alignment_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            {
                let perm_id = concat!(stringify!($C), "_", stringify!($G), "_", stringify!($N));
                let genomes: Vec<$G<$C>> = load_fixture(perm_id);
                let pairs: Vec<(Vec<$C>, Vec<$C>)> = genomes
                    .windows(2)
                    .map(|w| (w[0].connections().to_vec(), w[1].connections().to_vec()))
                    .collect();

                let id = concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N));
                $group.bench_function(
                    concat!("disjoint_excess/", stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter(|| {
                            pairs.iter().for_each(|(l, r)| {
                                criterion::black_box(disjoint_excess_count::<$C>(l, r));
                            })
                        })
                    },
                );
                let _ = id; // suppress unused warning
                $group.bench_function(
                    concat!("avg_param_diff/", stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter(|| {
                            pairs.iter().for_each(|(l, r)| {
                                criterion::black_box(avg_param_diff::<$C>(l, r));
                            })
                        })
                    },
                );
                $group.bench_function(
                    concat!("delta/", stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter(|| {
                            pairs.iter().for_each(|(l, r)| {
                                criterion::black_box(delta::<$C>(l, r));
                            })
                        })
                    },
                );
            }
        )*
    };
}

fn bench_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover");
    crossover_bench!(
        group,
        connections: [WConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );
    group.finish();
}

fn bench_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment");
    alignment_bench!(
        group,
        connections: [WConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );
    group.finish();
}

criterion_group!(benches, bench_crossover, bench_alignment);
criterion_main!(benches);
