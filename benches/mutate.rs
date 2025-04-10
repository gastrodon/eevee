use criterion::Criterion;
use eevee::{
    genome::InnoGen,
    genome::{Genome, Recurrent, WConnection},
    random::default_rng,
};

type C = WConnection;
type G = Recurrent<C>;

fn bench_mutate(bench: &mut Criterion) {
    let genome = G::from_str(include_str!("data/ctr-genome-rand-100.json")).unwrap();
    let mut rng = default_rng();
    bench.bench_function("mutate-connection", |b| {
        b.iter(|| {
            genome
                .clone()
                .new_connection(&mut rng, &mut InnoGen::new(300))
        })
    });

    bench.bench_function("mutate-bisection", |b| {
        b.iter(|| {
            genome
                .clone()
                .bisect_connection(&mut rng, &mut InnoGen::new(300))
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
        use core::time::Duration;
        Criterion::default()
            .measurement_time(Duration::from_millis(1))
            .sample_size(10)
            .nresamples(1)
            .without_plots()
            .configure_from_args()
    };
    bench_mutate(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
