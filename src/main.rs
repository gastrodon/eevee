#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

mod crossover;
mod genome;
mod network;
mod scenario;
mod specie;

use network::steep_sigmoid;
use rand::rng;
use scenario::{ConstXOR, EvolutionTarget, Scenario};
use specie::population_init;

const GENERATIONS: usize = 10;
const POPULATION: usize = 2500;

fn main() {
    let res = ConstXOR::new().evolve(
        EvolutionTarget::Generation(GENERATIONS),
        |(i, o)| population_init(i, o, POPULATION, &mut rng()),
        POPULATION,
        steep_sigmoid,
        0.22,
        0.4,
    );

    println!(
        "{:?}",
        res.0.iter().map(|(_, f)| f).take(10).collect::<Vec<_>>()
    );
}
