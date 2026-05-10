#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use core::{f64, ops::ControlFlow};
use eevee::{
    genome::{Genome, NonRecurrent, WConnection},
    network::{activate::steep_sigmoid, BinaryFeedForward, Network, ToNetwork},
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    Connection, EvolutionConfig, Scenario, Stats,
};
use rand::Rng;
use std::marker::PhantomData;

const POPULATION: usize = 25;

const XOR_PAIRS: [([f64; 2], f64); 4] = [
    ([0., 0.], 1.),
    ([1., 1.], 1.),
    ([1., 0.], -1.),
    ([0., 1.], -1.),
];

fn xor_training_data(n: usize, rng: &mut impl Rng) -> Vec<([f64; 2], f64)> {
    (0..n).map(|_| XOR_PAIRS[rng.random_range(0..4)]).collect()
}

struct Xor<NN: Network, C: Connection, G: Genome<C>> {
    _phantom: PhantomData<(NN, C, G)>,
}

impl<NN: Network, C: Connection, G: Genome<C>> Xor<NN, C, G> {
    fn new() -> Self {
        Xor {
            _phantom: PhantomData,
        }
    }
}

macro_rules! eval_pair {
    ($pair:expr, $want:expr, ($network:ident $fit:ident $σ:ident)) => {{
        $network.step(&$pair, $σ);
        // tanh maps unbounded y → (-1, 1); targets are ±1
        let v = $network.output()[0].tanh();
        let error = ($want - v).abs(); // max error is 2.0 (e.g. want=1, v≈-1)
        $fit += 1.0 - 0.5 * error; // scale so worst case = 0.0, perfect = 1.0
    }};
}

impl<NN: Network, C: Connection, G: Genome<C> + ToNetwork<NN, C>, A: Fn(f64) -> f64>
    Scenario<C, G, A> for Xor<NN, C, G>
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
        g.node_count(),
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

type NN = BinaryFeedForward;
type C = WConnection;
type G = NonRecurrent<C>;

fn main() {
    evolve(
        Xor::<NN, C, G>::new(),
        |(i, o)| population_init::<C, G>(i, o, POPULATION),
        steep_sigmoid,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook)]),
        EvolutionConfig::default(),
    );
}
