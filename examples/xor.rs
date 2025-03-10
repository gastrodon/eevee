#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{activate::relu, specie::population_init, EvolutionTarget, Network, Scenario, Specie};
use core::f64;

const POPULATION: usize = 100;

struct Xor;

impl Scenario for Xor {
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval<F: Fn(f64) -> f64>(&self, network: &mut impl Network, σ: F) -> f64 {
        let mut fit = 0.;
        network.step(2, &[0., 0.], &σ);
        fit += 1. - (1. - network.output()[0]).abs().powf(2.);
        network.flush();

        network.step(2, &[1., 1.], &σ);
        fit += 1. - (1. - network.output()[0]).abs().powf(2.);
        network.flush();

        network.step(2, &[0., 1.], &σ);
        fit += 1. - (0. - network.output()[0]).abs().powf(2.);
        network.flush();

        network.step(2, &[1., 0.], &σ);
        fit += 1. - (0. - network.output()[0]).abs().powf(2.);

        fit / 4.
    }
}

fn main() {
    let res = Xor {}.evolve(
        EvolutionTarget::Fitness(0.9999),
        |(i, o)| population_init(i, o, POPULATION),
        POPULATION,
        relu,
    );

    println!(
        "top score: {:?}",
        res.0
            .into_iter()
            .flat_map(|Specie { members, .. }| members)
            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
            .unwrap()
            .1
    );
}
