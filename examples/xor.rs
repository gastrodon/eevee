#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use core::ops::ControlFlow;
use eevee::{
    genome::{Recurrent, WConnection},
    network::{activate::steep_sigmoid, Realtime},
    playground::xor::{XorScenario, XOR_TARGET},
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    Connection, Genome, Stats,
};

const POPULATION: usize = 100;

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

    if stats.any_fitter_than(XOR_TARGET) {
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
        XorScenario::<Realtime>::default(),
        |(i, o)| population_init::<C, G>(i, o, POPULATION),
        steep_sigmoid,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
