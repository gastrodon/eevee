#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu,
    network::loss::decay_quadratic,
    random::{default_rng, Happens, ProbBinding, ProbStatic, Probabilities},
    specie::population_init,
    Ctrnn, EvolutionTarget, Genome, Network, Scenario, Specie,
};
use core::f64;
use rand::RngCore;

const POPULATION: usize = 100;

struct Xor;

impl<H: RngCore + Probabilities + Happens, A: Fn(f64) -> f64> Scenario<H, A> for Xor {
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&mut self, genome: &Genome, σ: &A) -> f64 {
        let mut network = Ctrnn::from_genome(genome);
        let mut fit = 0.;
        network.step(2, &[0., 0.], σ);
        fit += decay_quadratic(1., network.output()[0]);
        network.flush();

        network.step(2, &[1., 1.], σ);
        fit += decay_quadratic(1., network.output()[0]);
        network.flush();

        network.step(2, &[0., 1.], σ);
        fit += decay_quadratic(0., network.output()[0]);
        network.flush();

        network.step(2, &[1., 0.], σ);
        fit += decay_quadratic(0., network.output()[0]);

        fit / 4.
    }
}

fn main() {
    let res = Xor {}.evolve(
        EvolutionTarget::Fitness(0.749999),
        |(i, o)| population_init(i, o, POPULATION),
        POPULATION,
        &relu,
        &mut ProbBinding::new(ProbStatic::default(), default_rng()),
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
