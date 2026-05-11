use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    genome::{Recurrent, WConnection},
    network::{Continuous, FromGenome, NonBias},
    Connection, Genome, SerializeFile,
};
use eevee_macros::fn_matrix;
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

// ---------------------------------------------------------------------------
// fn_matrix! generates one build_<c>_<g>_<n> helper per (C, G, NN) triple.
//
// The TT-munching bench macro below calls them via paste! to compose the name
// from the same $C/$G/$N idents used to select the fixture directory, keeping
// the two in sync without duplication.
// ---------------------------------------------------------------------------

fn_matrix! {
    C: WConnection,
    G: Recurrent<WConnection>,
    NN: Continuous | NonBias,

    fn build(g: G) -> NN {
        NN::from_genome(&g)
    }
}

macro_rules! genome_to_nn_bench {
    (
        $group:expr,
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        genome_to_nn_bench!(@foreach_c $group, [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c $group:expr, [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c $group:expr, [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        genome_to_nn_bench!(@foreach_g $group, $C, $Gs, $Ns);
        genome_to_nn_bench!(@foreach_c $group, [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $group:expr, $C:ident, [], $Ns:tt) => {};
    (@foreach_g $group:expr, $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        genome_to_nn_bench!(@bench $group, $C, $G, $Ns);
        genome_to_nn_bench!(@foreach_g $group, $C, [$($Gs),*], $Ns);
    };

    (@bench $group:expr, $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            {
                let perm_id = concat!(stringify!($C), "_", stringify!($G), "_", stringify!($N));
                let genomes: Vec<$G<$C>> = load_fixture(perm_id);

                $group.bench_function(
                    concat!(stringify!($C), "/", stringify!($G), "/", stringify!($N)),
                    |b| {
                        b.iter_batched(
                            || genomes[0].clone(),
                            // Calls the fn_matrix!-generated helper:
                            //   build_wconnection_recurrent_continuous / _nonbias
                            |g| paste::paste! {
                                [<build_ $C:lower _ $G:lower _ $N:lower>](g)
                            },
                            BatchSize::SmallInput,
                        )
                    },
                );
            }
        )*
    };
}

fn bench_genome_to_nn(c: &mut Criterion) {
    let mut group = c.benchmark_group("genome_to_nn");
    genome_to_nn_bench!(
        group,
        connections: [WConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );
    group.finish();
}

criterion_group!(benches, bench_genome_to_nn);
criterion_main!(benches);
