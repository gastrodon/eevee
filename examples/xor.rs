use core::f64;

use brain::{
    activate::{relu, steep_sigmoid},
    specie::population_init,
    EvolutionTarget, Network, Scenario,
};
use rand::rng;

const POPULATION: usize = 2500;

pub struct Xor;

impl Scenario for Xor {
    fn io() -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, network: &mut impl Network) -> f64 {
        let mut fit = 0.;
        network.step(2, &[0., 0.]);
        fit += 1. - (1. - network.output()[0]).abs().powf(2.);

        network.step(2, &[1., 1.]);
        fit += 1. - (1. - network.output()[0]).abs().powf(2.);

        network.step(2, &[0., 1.]);
        fit += 1. - (0. - network.output()[0]).abs().powf(2.);

        network.step(2, &[1., 0.]);
        fit += 1. - (0. - network.output()[0]).abs().powf(2.);

        fit / 4.
    }
}

fn main() {
    let mut res = Xor {}.evolve(
        // EvolutionTarget::Fitness(scale(0) as usize),
        EvolutionTarget::Fitness(0.99),
        |(i, o)| population_init(i, o, POPULATION, &mut rng()),
        POPULATION,
        steep_sigmoid,
        0.22,
        0.4,
    );

    res.0.sort_by(|(_, l), (_, r)| r.partial_cmp(l).unwrap());
    let res = Xor {}.evolve(
        // EvolutionTarget::Fitness(scale(0) as usize),
        EvolutionTarget::Fitness(0.99999),
        |_| (res.0.into_iter().map(|(g, _)| g).take(100).collect(), res.1),
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
