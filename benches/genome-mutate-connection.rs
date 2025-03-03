use brain::{specie::InnoGen, Genome};
use criterion::Criterion;
use rand::rng;

fn bench(bench: &mut Criterion) {
    let genome = Genome::from_str(include_str!("genome-rand-100.json")).unwrap();
    bench.bench_function("genome-mutate-connection", |b| {
        b.iter(|| {
            genome
                .clone()
                .mutate_connection(&mut rng(), &mut InnoGen::new(300))
                .unwrap()
        })
    });
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(2000)
        .significance_level(0.1);
    #[cfg(feature = "smol_bench")]
    let mut criterion: criterion::Criterion<_> = {
        use std::time::Duration;
        Criterion::default()
            .measurement_time(Duration::from_millis(1))
            .sample_size(10)
            .nresamples(1)
            .without_plots()
            .configure_from_args()
    };
    bench(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
