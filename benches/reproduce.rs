use brain::{
    genome::{Connection, Genome, InnoGen, Recurrent, WConnection},
    random::default_rng,
    reproduce::reproduce,
};
use criterion::Criterion;

type C = WConnection;
type G = Recurrent<C>;

fn bench_reproduce(bench: &mut Criterion) {
    let genomes =
        serde_json::from_str::<Vec<(G, _)>>(include_str!("data/ctr-genome-xor-100.json")).unwrap();
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

    let mut rng = default_rng();
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
