use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    crossover::delta,
    genome::{Recurrent, WConnection},
    population::{speciate, SpecieRepr},
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

// ---------------------------------------------------------------------------
// speciate: full population speciation
// ---------------------------------------------------------------------------
//
// Uses every 10th genome as a specie representative, matching a realistic
// scenario where reprs come from the previous generation.

fn bench_speciate(c: &mut Criterion) {
    let mut group = c.benchmark_group("speciate");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Continuous | NonBias,
        {
            let genomes: Vec<G> = load_fixture(PERM_ID);
            let reprs: Vec<SpecieRepr<C>> = genomes
                .iter()
                .step_by(10)
                .map(|g| SpecieRepr::new(g.connections().to_vec()))
                .collect();
            group.bench_function(BENCH_ID, |b| {
                b.iter_batched(
                    || {
                        let fitted: Vec<(G, f64)> = genomes
                            .iter()
                            .enumerate()
                            .map(|(i, g)| (g.clone(), i as f64))
                            .collect();
                        (fitted, reprs.clone())
                    },
                    |(fitted, reprs)| speciate(fitted.into_iter(), reprs.into_iter()),
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// delta: pairwise distance between connection slices — the inner hot path of
// speciate (called once per genome per specie repr).
// ---------------------------------------------------------------------------

fn bench_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Continuous | NonBias,
        {
            let genomes: Vec<G> = load_fixture(PERM_ID);
            let pairs: Vec<(Vec<C>, Vec<C>)> = genomes
                .windows(2)
                .map(|w| (w[0].connections().to_vec(), w[1].connections().to_vec()))
                .collect();
            group.bench_function(BENCH_ID, |b| {
                b.iter(|| {
                    pairs.iter().for_each(|(l, r)| {
                        criterion::black_box(delta::<C>(l, r));
                    })
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_speciate, bench_delta);
criterion_main!(benches);
