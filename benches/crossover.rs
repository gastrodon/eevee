use core::cmp::Ordering;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    crossover::{avg_param_diff, crossover, delta, disjoint_excess_count},
    genome::{Recurrent, WConnection},
    random::default_rng,
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
// crossover: merge two connection slices via NEAT crossover.
// Allocates a new Vec<C> each call — allocation cost is intentionally included.
// ---------------------------------------------------------------------------

fn bench_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Continuous | NonBias,
        {
            let genomes: Vec<G> = load_fixture(PERM_ID);
            let l = genomes[0].connections().to_vec();
            let r = genomes[1 % genomes.len()].connections().to_vec();
            let mut rng = default_rng();
            group.bench_function(BENCH_ID, |b| {
                b.iter_batched(
                    || (l.clone(), r.clone()),
                    |(l, r)| crossover(&l, &r, Ordering::Greater, &mut rng),
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// disjoint_excess_count: sorted merge scan counting misaligned innovations.
// Pure read — no allocation.
// ---------------------------------------------------------------------------
fn bench_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Continuous | NonBias,
        {
            let bench_id = BENCH_ID;
            let genomes: Vec<G> = load_fixture(PERM_ID);
            let pairs: Vec<(Vec<C>, Vec<C>)> = genomes
                .windows(2)
                .map(|w| (w[0].connections().to_vec(), w[1].connections().to_vec()))
                .collect();

            group.bench_function(format!("disjoint_excess/{bench_id}"), |b| {
                b.iter(|| {
                    pairs.iter().for_each(|(l, r)| {
                        criterion::black_box(disjoint_excess_count::<C>(l, r));
                    })
                })
            });
            group.bench_function(format!("avg_param_diff/{bench_id}"), |b| {
                b.iter(|| {
                    pairs.iter().for_each(|(l, r)| {
                        criterion::black_box(avg_param_diff::<C>(l, r));
                    })
                })
            });
            group.bench_function(format!("delta/{bench_id}"), |b| {
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

criterion_group!(benches, bench_crossover, bench_alignment);
criterion_main!(benches);
