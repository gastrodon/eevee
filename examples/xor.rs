#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use approx::relative_eq;
use core::{f64, ops::ControlFlow};
use eevee::{
    activate::relu,
    genome::{Genome, Recurrent, WConnection},
    network::{Network, Simple, ToNetwork},
    population::population_init_diverse,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    Connection, Scenario, Stats,
};

const POPULATION: usize = 1000;

struct Xor;

macro_rules! eval_pair {
    ($pair:expr, $want:expr, ($network:ident $fit:ident $σ:ident)) => {{
        $network.step(2, &$pair, $σ);
        let v = $network.output()[0];
        if relative_eq!(v, $want, epsilon = 0.05) {
            $fit += 100.;
        } else if (-1. ..=2.).contains(&v) {
            $fit -= ($want - v).abs();
        } else {
            $fit -= v.abs() * v.abs();
        }
        $network.flush();
    }};
}

impl<C: Connection, G: Genome<C> + ToNetwork<Simple<C>, C>, A: Fn(f64) -> f64> Scenario<C, G, A>
    for Xor
{
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = genome.network();
        let mut fit = 0.;

        eval_pair!([0., 0.], 1., (network fit σ));
        eval_pair!([1., 1.], 1., (network fit σ));
        eval_pair!([1., 0.], 0., (network fit σ));
        eval_pair!([0., 1.], 0., (network fit σ));

        fit
    }
}

fn hook<C: Connection, G: Genome<C>>(stats: &mut Stats<'_, C, G>) -> ControlFlow<()> {
    if stats.generation % 100 == 1 {
        let (_, f) = stats.fittest().unwrap();
        println!(
            "fittest of gen {}: {:.4} (of {} species",
            stats.generation,
            f,
            stats.species.len()
        );
    }

    if stats.any_fitter_than(400. - f64::EPSILON) {
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
        |(i, o)| {
            let mut rng = default_rng();
            population_init_diverse::<C, G>(i, o, POPULATION, &mut rng)
        },
        relu,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
