#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{activate::relu, specie::population_init, EvolutionTarget, Network, Scenario};
use core::f64;
use rand::rng;

const POPULATION: usize = 2500;

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

fn main() {
    let res = Xor {}.evolve(
        EvolutionTarget::Fitness(0.9999),
        |(i, o)| population_init(i, o, POPULATION, &mut rng()),
        POPULATION,
        relu,
        0.22,
        0.4,
    );

    println!(
        "top score: {:?}",
        res.0
            .into_iter()
            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
            .unwrap()
            .1
    );
}
