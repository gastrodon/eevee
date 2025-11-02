use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_seed(c: &mut Criterion) {
    c.bench_function("seed", |b| b.iter(|| black_box(eevee::random::default_rng())));
}

criterion_group!(benches, bench_seed);
criterion_main!(benches);