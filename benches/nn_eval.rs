use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    activate::steep_sigmoid,
    genome::{Recurrent, WConnection},
    network::{FromGenome, Realtime},
    Connection, Genome, Network, SerializeFile,
};
use eevee_macros::fn_matrix;
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

fn bench_nn_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("nn_step");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Realtime,
        {
            let genomes: Vec<G> = load_fixture(PERM_ID);
            // Use the first (simplest) genome for a stable baseline.
            // Genomes are sorted by file name, so genome 0 is deterministic.
            let genome = genomes.first().expect("fixture is empty");
            group.bench_function(BENCH_ID, |b| {
                b.iter_batched(
                    || <NN as FromGenome<C, G>>::from_genome(genome),
                    |mut nn| {
                        nn.step(&[1.0_f64, 0.0_f64], steep_sigmoid);
                        nn
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_nn_step);
criterion_main!(benches);
