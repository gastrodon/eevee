use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use eevee::{
    genome::{InnoGen, Recurrent, WConnection},
    population::speciate,
    random::default_rng,
    reproduce::{population_reproduce, reproduce},
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

fn bench_mutate(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutate");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Continuous | NonBias,
        {
            let genomes: Vec<G> = load_fixture(PERM_ID);
            let head = inno_head::<C, G>(&genomes);
            group.bench_function(BENCH_ID, |b| {
                b.iter_batched(
                    || (genomes[0].clone(), InnoGen::new(head), default_rng()),
                    |(mut g, mut inno, mut rng)| {
                        let _ = g.mutate(&mut rng, &mut inno);
                        (g, inno, rng)
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// reproduce: elite + copy + crossover for a single specie.
// Fitness is assigned by index so the fittest genome is deterministic.
// ---------------------------------------------------------------------------

fn bench_reproduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("reproduce");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Continuous | NonBias,
        {
            let genomes: Vec<G> = load_fixture(PERM_ID);
            let head = inno_head::<C, G>(&genomes);
            let n = genomes.len();
            group.bench_function(BENCH_ID, |b| {
                b.iter_batched(
                    || {
                        let members: Vec<(G, f64)> = genomes
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
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// population_reproduce: full speciated-population reproduction pipeline.
// Species are constructed once outside the measurement.
// ---------------------------------------------------------------------------

fn bench_population_reproduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("population_reproduce");
    fn_matrix! {
        C: WConnection,
        G: Recurrent<WConnection>,
        NN: Continuous | NonBias,
        {
            let genomes: Vec<G> = load_fixture(PERM_ID);
            let head = inno_head::<C, G>(&genomes);
            let n = genomes.len();
            let species = speciate(
                genomes.iter().enumerate().map(|(i, g)| (g.clone(), i as f64)),
                std::iter::empty(),
            );
            group.bench_function(BENCH_ID, |b| {
                b.iter_batched(
                    || default_rng(),
                    |mut rng| population_reproduce(&species, n, head, &mut rng),
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_mutate,
    bench_reproduce,
    bench_population_reproduce
);
criterion_main!(benches);
