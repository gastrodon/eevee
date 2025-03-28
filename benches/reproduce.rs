use brain::{
    genome::{CTRGenome, Connection, Genome},
    random::{default_rng, ProbBinding, ProbStatic},
    specie::{reproduce, InnoGen},
};
use criterion::Criterion;

fn bench_reproduce(bench: &mut Criterion) {
    let genomes =
        serde_json::from_str::<Vec<(CTRGenome, _)>>(include_str!("data/ctr-genome-xor-100.json"))
            .unwrap();
    let inno_head = genomes
        .iter()
        .map(|(genome, _)| {
            genome
                .connections()
                .iter()
                .map(|connection| connection.inno())
                .max()
                .unwrap()
        })
        .max()
        .unwrap();

    let mut rng = ProbBinding::new(ProbStatic::default(), default_rng());
    bench.bench_function("reproduce", |b| {
        b.iter(|| reproduce(genomes.clone(), 100, &mut InnoGen::new(inno_head), &mut rng))
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
    bench_reproduce(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
