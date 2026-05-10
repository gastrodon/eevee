use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    activate::steep_sigmoid,
    genome::{Recurrent, WConnection},
    network::{Continuous, FromGenome, NonBias},
    Connection, Genome, Network, SerializeFile,
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
// nn_step: Network::step on a trained XOR network.
//
// Network construction is in the setup — only step() itself is timed.
// The network is returned from the bench closure so its drop falls outside
// the measurement window (criterion iter_batched guarantee).
// ---------------------------------------------------------------------------

macro_rules! nn_step_bench {
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        nn_step_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        nn_step_bench!(@foreach_g $group, $C, $Gs, $Ns);
        nn_step_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        nn_step_bench!(@bench $group, $C, $G, $Ns);
        nn_step_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            {
                let perm_id = concat!(stringify!($C), "_", stringify!($G), "_", stringify!($N));
                let genomes: Vec<$G<$C>> = load_fixture(perm_id);
                // Use the first (simplest) genome for a stable baseline.
                // Genomes are sorted by file name, so genome 0 is deterministic.
                let genome = genomes.first().expect("fixture is empty");

                $group.bench_function(
                    concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter_batched(
                            || <$N as FromGenome<$C, $G<$C>>>::from_genome(genome),
                            |mut nn| {
                                nn.step(20, &[1.0_f64, 0.0_f64], steep_sigmoid);
                                nn
                            },
                            BatchSize::SmallInput,
                        )
                    },
                );
            }
        )*
    };
}

fn bench_nn_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("nn_step");
    nn_step_bench!(
        group,
        connections: [WConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );
    group.finish();
}

criterion_group!(benches, bench_nn_step);
criterion_main!(benches);
