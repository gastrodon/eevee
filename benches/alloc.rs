use brain::{
    genome::{Recurrent, WConnection},
    population::speciate,
    reproduce::population_alloc,
};
use criterion::Criterion;

type C = WConnection;
type G = Recurrent<C>;

fn bench_alloc(bench: &mut Criterion) {
    let population = 100;
    let species = speciate(
        serde_json::from_str::<Vec<(G, _)>>(include_str!("data/ctr-genome-xor-100.json"))
            .unwrap()
            .into_iter(),
        vec![].into_iter(),
    );

    bench.bench_function("alloc", |b| {
        b.iter(|| population_alloc(species.iter(), population))
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
    bench_alloc(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
