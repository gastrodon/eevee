#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use criterion::Criterion;
use eevee::{activate::relu, network::Realtime, Network as _, SerializeFile as _};

fn bench_nn(bench: &mut Criterion) {
    let mut net = Realtime::from_str(include_str!("data/ctrnn-rand-100.json")).unwrap();
    net.prec = 100;
    let i = vec![0.7, 0.3];

    bench.bench_function("ctrnn-step", |b| {
        b.iter_batched(
            || net.clone(),
            |mut net| net.step(&i, relu),
            criterion::BatchSize::SmallInput,
        )
    });
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(1000)
        .significance_level(0.1);
    #[cfg(feature = "smol_bench")]
    let mut criterion: criterion::Criterion<_> = {
        use core::time::Duration;
        Criterion::default()
            .measurement_time(Duration::from_millis(1))
            .sample_size(10)
            .nresamples(1)
            .without_plots()
            .configure_from_args()
    };
    bench_nn(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
