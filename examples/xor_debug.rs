#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use approx::relative_eq;
use core::{f64, ops::ControlFlow};
use eevee::{
    activate::relu,
    genome::{Genome, Recurrent, WConnection},
    network::{Network, Simple, ToNetwork},
    population::population_init,
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

fn detailed_eval<C: Connection, G: Genome<C> + ToNetwork<Simple<C>, C>>(genome: &G) {
    let mut network = genome.network();
    let relu = |x: f64| if x > 0. { x } else { 0. };
    
    println!("\nDetailed evaluation of best genome:");
    println!("Connections: {}", genome.connections().len());
    
    for test_case in [([0., 0.], 1.), ([1., 1.], 1.), ([1., 0.], 0.), ([0., 1.], 0.)] {
        let (inputs, expected) = test_case;
        network.step(2, &inputs, &relu);
        let output = network.output()[0];
        let correct = relative_eq!(output, expected, epsilon = 0.05);
        println!("  Input {:?} -> Output: {:.4}, Expected: {}, Correct: {}", 
                 inputs, output, expected, correct);
        network.flush();
    }
}

fn hook<C: Connection, G: Genome<C> + ToNetwork<Simple<C>, C>>(
    stats: &mut Stats<'_, C, G>
) -> ControlFlow<()> {
    if stats.generation % 50 == 0 || stats.generation < 5 {
        let (genome, f) = stats.fittest().unwrap();
        println!(
            "Gen {}: best_fitness={:.2}, species={}, connections={}",
            stats.generation,
            f,
            stats.species.len(),
            genome.connections().len()
        );
    }

    if stats.generation == 100 {
        let (genome, _) = stats.fittest().unwrap();
        detailed_eval(genome);
        return ControlFlow::Break(());
    }

    if stats.any_fitter_than(400. - f64::EPSILON) {
        let (genome, f) = stats.fittest().unwrap();
        println!("\n✓ XOR SOLVED! Gen {}: fitness={:.4}", stats.generation, f);
        detailed_eval(genome);
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
