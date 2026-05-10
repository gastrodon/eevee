use criterion::Criterion;
use eevee::{genome::InnoGen, random::default_rng};
use rand::Rng;

fn bench_innogen(bench: &mut Criterion) {
    let mut rng = default_rng();
    // Pre-generate pairs so rng cost is outside the hot path and each
    // iteration sees the same sequence on a fresh InnoGen.
    let pairs: Vec<(usize, usize)> = (0..100)
        .map(|_| (rng.random_range(0..=10_000), rng.random_range(0..=10_000)))
        .collect();

    bench.bench_function("innogen", |b| {
        b.iter_batched(
            || InnoGen::new(0),
            |mut inno| {
                pairs.iter().for_each(|&p| {
                    inno.path(p);
                })
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(2000)
        .significance_level(0.1)
        .configure_from_args();
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
    bench_innogen(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
