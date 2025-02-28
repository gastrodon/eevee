#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

mod crossover;
mod genome;
mod network;
mod scenario;
mod specie;

use network::steep_sigmoid;
use rand::rng;
use scenario::{ConstXOR, Scenario};
use specie::{population_init, population_reproduce, speciate};

const GENERATIONS: usize = 100_000;
const POPULATION: usize = 7500;

fn main() {
    let mut rng = rng();
    let (mut population, mut inno_head) = population_init(1, 2, POPULATION, &mut rng);
    let mut gen_idx = 0;
    let game = ConstXOR::new();

    let pop_evaluated = loop {
        let scored = population
            .iter()
            .map(|genome| (genome, game.eval(&mut genome.network(steep_sigmoid))))
            .collect::<Vec<_>>();

        let species = {
            let mut species = speciate(scored.into_iter());
            for s in species.iter_mut() {
                s.shrink_top_p(0.1 / 3.);
            }
            species
        };

        if gen_idx == GENERATIONS {
            break species;
        }

        if gen_idx % 100 == 0 {
            for (idx, s) in species.iter().filter(|s| !s.is_empty()).enumerate() {
                println!("champ {gen_idx}.{idx}: {}", s.1[0].1)
            }
        }
        (population, inno_head) =
            population_reproduce(&species, POPULATION, 0.4, inno_head, &mut rng);

        gen_idx += 1;
    };

    println!("{gen_idx}");
    for (idx, s) in pop_evaluated.iter().filter(|s| !s.is_empty()).enumerate() {
        println!("champ {idx}: {}", s.1[0].1)
    }
}
