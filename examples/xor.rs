#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu,
    network::loss::decay_quadratic,
    random::{
        default_rng, percent, EvolutionEvent, Happens, ProbBinding, ProbStatic, Probabilities,
    },
    scenario::EvolutionHooks,
    specie::population_init,
    Ctrnn, Genome, Network, Scenario, Specie,
};
use core::{f64, ops::ControlFlow};
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
    Xor {}.evolve(
        |(i, o)| population_init(i, o, POPULATION),
        POPULATION,
        &relu,
        &mut ProbBinding::new(ProbStatic::default(), default_rng()),
        EvolutionHooks::new(vec![
            Box::new(|stats| {
                if stats.any_fitter_than(0.749999) {
                    println!("gen: {}", stats.generation);
                    println!(
                        "top score: {:?}",
                        stats
                            .species
                            .iter()
                            .flat_map(|Specie { members, .. }| members)
                            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
                            .unwrap()
                            .1
                    );
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            }),
            Box::new(|stats| {
                if stats.generation == 100 {
                    stats
                        .rng
                        .update((EvolutionEvent::MutateConnection, percent(35)));
                    stats
                        .rng
                        .update((EvolutionEvent::MutateBisection, percent(35)));
                }

                ControlFlow::Continue(())
            }),
            Box::new(|stats| {
                if stats.generation % 10 == 1 {
                    let (_, f) = stats.fittest().unwrap();
                    println!("fittest of gen {}: {:.4}", stats.generation, f);
                }

                ControlFlow::Continue(())
            }),
        ]),
    );
}
