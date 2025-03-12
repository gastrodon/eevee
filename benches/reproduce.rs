use brain::{
    specie::{reproduce, InnoGen},
    Connection, Genome,
};
use criterion::Criterion;
use rand::rng;

fn bench(bench: &mut Criterion) {
    let genomes = serde_json::from_str::<Vec<_>>(include_str!("data/genome-xor-100.json")).unwrap();
    let inno_head = *genomes
        .iter()
        .map(|(Genome { connections, .. }, _)| {
            connections
                .iter()
                .map(|Connection { inno, .. }| inno)
                .max()
                .unwrap()
        })
        .max()
        .unwrap();

    bench.bench_function("reproduce", |b| {
        b.iter(|| {
            reproduce(
                genomes.clone(),
                100,
                &mut InnoGen::new(inno_head),
                &mut rng(),
            )
        })
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
    bench(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
