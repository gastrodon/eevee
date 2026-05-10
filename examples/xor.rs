#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use core::{f64, ops::ControlFlow};
use eevee::{
    genome::{Genome, Recurrent, WConnection},
    network::{activate::steep_sigmoid, Continuous, Network, ToNetwork},
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    Connection, Scenario, Stats,
};
use rand::Rng;

const POPULATION: usize = 100;

const XOR_PAIRS: [([f64; 2], f64); 4] = [
    ([0., 0.], 1.),
    ([1., 1.], 1.),
    ([1., 0.], -1.),
    ([0., 1.], -1.),
];

fn xor_training_data(n: usize, rng: &mut impl Rng) -> Vec<([f64; 2], f64)> {
    (0..n).map(|_| XOR_PAIRS[rng.random_range(0..4)]).collect()
}

struct Xor;

macro_rules! eval_pair {
    ($pair:expr, $want:expr, ($network:ident $fit:ident $σ:ident)) => {{
        $network.step(20, &$pair, $σ);
        // tanh maps unbounded y → (-1, 1); targets are ±1
        let v = $network.output()[0].tanh();
        let error = ($want - v).abs(); // max error is 2.0 (e.g. want=1, v≈-1)
        $fit += 1.0 - 0.5 * error;    // scale so worst case = 0.0, perfect = 1.0
        $network.flush();
    }};
}

impl<C: Connection, G: Genome<C> + ToNetwork<Continuous, C>, A: Fn(f64) -> f64> Scenario<C, G, A>
    for Xor
{
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = genome.network();
        let mut fit = 0.;

        for (input, want) in xor_training_data(10, &mut rand::rng()) {
            eval_pair!(input, want, (network fit σ));
        }

        fit
    }
}

fn hook<C: Connection, G: Genome<C>>(stats: &mut Stats<'_, C, G>) -> ControlFlow<()> {
    let (g, f) = stats.fittest().unwrap();
    let total = stats.species.iter().map(|s| s.len()).sum::<usize>() as f64;
    let breakdown = stats
        .species
        .iter()
        .map(|s| format!("{:.0}%", 100. * s.len() as f64 / total))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "gen {}: {:.4} ({} nodes, {} conns) of {} species [{}]",
        stats.generation,
        f,
        g.nodes().len(),
        g.connections().len(),
        stats.species.len(),
        breakdown,
    );

    if stats.any_fitter_than(9.5) {
        println!("target met in gen {}", stats.generation);
        return ControlFlow::Break(());
    }

    if stats.generation >= 200 {
        println!("generation limit reached");
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
        steep_sigmoid,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
