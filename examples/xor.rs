#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu,
    genome::{Genome, Recurrent, WConnection},
    network::{loss::decay_quadratic, Continuous, Network, ToNetwork},
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    specie::population_init,
    Connection, Scenario, Stats,
};
use core::{f64, ops::ControlFlow};

const POPULATION: usize = 100;

struct Xor;

impl<C: Connection, G: Genome<C> + ToNetwork<Continuous, C>, A: Fn(f64) -> f64> Scenario<C, G, A>
    for Xor
{
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = genome.network();
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

fn hook<C: Connection, G: Genome<C>>(stats: &mut Stats<'_, C, G>) -> ControlFlow<()> {
    if stats.generation % 10 == 1 {
        let (_, f) = stats.fittest().unwrap();
        println!(
            "fittest of gen {}: {:.4} (of {} species",
            stats.generation,
            f,
            stats.species.len()
        );
    }

    if stats.any_fitter_than(0.749999) {
        let fittest = stats.fittest().unwrap();
        println!("target met in gen {}: {:.4}", stats.generation, fittest.1);
        fittest
            .0
            .to_file(format!("output/xor-{}.json", stats.generation))
            .unwrap();

        return ControlFlow::Break(());
    }

    ControlFlow::Continue(())
}

type C = WConnection;
type G = Recurrent<C>;

fn main() {
    evolve(
        Xor {},
        |(i, o)| population_init::<C, G>(i, o, POPULATION),
        relu,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
