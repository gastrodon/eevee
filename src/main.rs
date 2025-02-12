mod crossover;
mod eval;
mod genome;
mod specie;

use eval::GameXOR;
use rand::{rng, rngs::ThreadRng};
use specie::{speciate, InnoGen, Specie, SpecieRepr};
use std::collections::HashMap;

use crate::genome::Genome;

const POPULATION: usize = 100;
const FIT_SCALE: usize = 512;

fn population_init(
    sensory: usize,
    action: usize,
    population: usize,
    rng: &mut ThreadRng,
) -> Vec<Genome> {
    let mut v = vec![Genome::new(sensory, action); population];
    let mut inext = InnoGen::new(0);
    for g in v.iter_mut() {
        g.mutate_connection(rng, &mut inext).unwrap();
    }
    v
}

fn specie_populations<'a>(
    species: &'a Vec<Specie<'a>>,
    population: usize,
) -> HashMap<&'a SpecieRepr<'a>, usize> {
    let fits = species
        .iter()
        .map(|s| (&s.0, s.fit_adjusted()))
        .collect::<HashMap<_, _>>();

    let population = population as f64;
    let fit_total = fits.values().sum::<f64>();
    fits.into_iter()
        .map(|(s, f)| (s, (population * (f / fit_total)) as usize))
        .collect()
}

fn main() {
    // start: rng, empty population
    // iter: evaluate population, bind fit,

    let mut rng = rng();
    let mut inno_head = 0;

    let mut population = population_init(2, 1, POPULATION, &mut rng);
    for gen_idx in 0..8 {
        println!("gen {gen_idx}...");

        let scored = population
            .iter()
            .map(|genome| {
                let fit = genome.propagate_game(&mut GameXOR::new(FIT_SCALE), true);
                (genome, fit / FIT_SCALE)
            })
            .collect::<Vec<_>>();

        let species = speciate(scored.into_iter());
        let species_pop = specie_populations(&species, POPULATION);

        //
        dbg!(species
            .iter()
            .map(|specie| (specie.fit_adjusted(), species_pop.get(&specie.0).unwrap()))
            .collect::<Vec<_>>());
        //

        let mut innogen = InnoGen::new(inno_head);
        population = species
            .iter()
            .flat_map(|specie| {
                specie
                    .reproduce(*species_pop.get(&specie.0).unwrap(), &mut innogen, &mut rng)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        inno_head = dbg!(innogen.head);
    }
}
