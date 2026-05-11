use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    genome::{Recurrent, WConnection},
    network::{Continuous, FromGenome, NonBias},
    Connection, Genome, SerializeFile,
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

fn bench_genome_to_nn(c: &mut Criterion) {
    let mut group = c.benchmark_group("genome_to_nn");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Continuous | NonBias,
        {
            let genomes: Vec<G> = load_fixture(PERM_ID);
            group.bench_function(BENCH_ID, |b| {
                b.iter_batched(
                    || genomes[0].clone(),
                    |g| NN::from_genome(&g),
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_genome_to_nn);
criterion_main!(benches);
