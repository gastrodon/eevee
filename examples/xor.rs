#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu, network::loss::decay_quadratic, specie::population_init, Ctrnn,
    EvolutionTarget, Genome, Network, Scenario, Specie,
};
use core::f64;

const POPULATION: usize = 100;

struct Xor;

impl Scenario for Xor {
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval<F: Fn(f64) -> f64>(&mut self, genome: &Genome, σ: F) -> f64 {
        let mut network = Ctrnn::from_genome(genome);
        let mut fit = 0.;
        network.step(2, &[0., 0.], &σ);
        fit += decay_quadratic(1., network.output()[0]);
        network.flush();

        network.step(2, &[1., 1.], &σ);
        fit += decay_quadratic(1., network.output()[0]);
        network.flush();

        network.step(2, &[0., 1.], &σ);
        fit += decay_quadratic(0., network.output()[0]);
        network.flush();

        network.step(2, &[1., 0.], &σ);
        fit += decay_quadratic(0., network.output()[0]);

        fit / 4.
    }
}

fn main() {
    let res = Xor {}.evolve(
        EvolutionTarget::Fitness(0.749999),
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
