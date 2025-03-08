#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{activate::relu, specie::population_init, EvolutionTarget, Network, Scenario};
use core::f64;
use criterion::{BatchSize, Criterion};
use rand::rng;

const POPULATION: usize = 100;

struct Xor;

impl Scenario for Xor {
    fn io() -> (usize, usize) {
        (2, 1)
    }

    fn eval<F: Fn(f64) -> f64>(&self, network: &mut impl Network, σ: F) -> f64 {
        let mut fit = 0.;
        network.step(2, &[0., 0.], &σ);
        fit += 1. - (1. - network.output()[0]).abs().powf(2.);

        network.step(2, &[1., 1.], &σ);
        fit += 1. - (1. - network.output()[0]).abs().powf(2.);

        network.step(2, &[0., 1.], &σ);
        fit += 1. - (0. - network.output()[0]).abs().powf(2.);

        network.step(2, &[1., 0.], &σ);
        fit += 1. - (0. - network.output()[0]).abs().powf(2.);

        fit / 4.
    }
}

fn bench(bench: &mut Criterion) {
    bench.bench_function("xor-100", |b| {
        b.iter_batched(
            || (),
            |_| {
                Xor {}.evolve(
                    EvolutionTarget::Generation(100),
                    |(i, o)| population_init(i, o, POPULATION, &mut rng()),
                    POPULATION,
                    relu,
                    0.4,
                    0.22,
                )
            },
            BatchSize::NumIterations(1),
        )
    });
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(100)
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
