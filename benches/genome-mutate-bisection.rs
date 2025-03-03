use brain::{specie::InnoGen, Genome};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::rng;

fn bench(bench: &mut Criterion) {
    let genome = Genome::from_str(include_str!("genome-rand-100.json")).unwrap();
    bench.bench_function("genome-mutate-bisection", |b| {
        b.iter(|| {
            genome
                .clone()
                .mutate_bisection(&mut rng(), &mut InnoGen::new(300))
                .unwrap()
        })
    });
}

criterion_group!(
  name = benches;
  config = Criterion::default().sample_size(50).significance_level(0.1);
  targets = bench
);
criterion_main!(benches);
