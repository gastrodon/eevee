use brain::{specie::InnoGen, Genome};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rng;

fn bench(bench: &mut Criterion) {
    let genome = Genome::from_str(include_str!("genome-rand-100.json")).unwrap();
    let head = genome
        .connections
        .iter()
        .max_by_key(|c| c.inno)
        .unwrap()
        .inno
        + 1;

    bench.bench_function("genome-mutate_bisection", |b| {
        b.iter(|| {
            genome
                .clone()
                .mutate_bisection(&mut rng(), &mut InnoGen::new(head))
        })
    });

    bench.bench_function("genome-mutate_connection", |b| {
        b.iter(|| {
            genome
                .clone()
                .mutate_bisection(&mut rng(), &mut InnoGen::new(head))
        })
    });
}

criterion_group!(
  name = benches;
  config = Criterion::default();
  targets = bench
);
criterion_main!(benches);
