use brain::specie::speciate;
use criterion::Criterion;
fn bench(bench: &mut Criterion) {
    let genomes = serde_json::from_str::<Vec<_>>(include_str!("data/genome-xor-100.json")).unwrap();
    bench.bench_function("speciate", |b| b.iter(|| speciate(genomes.clone(), vec![])));
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(1000)
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
