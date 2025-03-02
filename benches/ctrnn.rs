#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use std::time::Duration;

use brain::{activate::relu, Ctrnn, Network};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench(bench: &mut Criterion) {
    let net = &mut Ctrnn::from_str(include_str!("ctrnn-rand-100.json")).unwrap();
    let i = vec![0.7, 0.3];

    bench.bench_function("ctrnn-step", |b| b.iter(|| net.step(100, &i, relu)));
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench
);
criterion_main!(benches);
