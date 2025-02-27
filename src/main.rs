#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

mod crossover;
mod ctrnn;
mod eval;
mod genome;
mod specie;

use eval::{steep_sigmoid, Game, GameXOR};
use genome::Genome;
use rand::{rng, rngs::ThreadRng};
use specie::{speciate, InnoGen, Specie, SpecieRepr};
use std::collections::HashMap;

const GENERATIONS: usize = 100_000;
const POPULATION: usize = 7500;

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

fn specie_sizer(population: usize, top_p: f64) -> Box<dyn Fn(f64, f64) -> usize> {
    let pop_adj = population as f64 / top_p;
    Box::new(move |fit_1, fit_total| (pop_adj * fit_1 / fit_total).round() as usize)
}

// drop_p is a whole percent
fn specie_populations<'a>(
    species: &'a Vec<Specie<'a>>,
    population: usize,
    top_p: f64,
) -> HashMap<&'a SpecieRepr<'a>, usize> {
    let mut fits = species
        .iter()
        .map(|s| (&s.0, s.fit_adjusted()))
        .collect::<Vec<_>>();

    // I speculate partial_cmp may be none if the fitness is NaN,
    // which would indicate a bigger issue somewhere else
    fits.sort_by(|(_, l), (_, r)| r.partial_cmp(l).unwrap());

    let sizer = specie_sizer(population, top_p);
    let fit_total = fits.iter().fold(0., |acc, (_, n)| acc + n);

    let mut sizes = HashMap::new();
    let mut size_acc = 0;
    for (s_repr, fit) in fits.into_iter() {
        let s_size = sizer(fit, fit_total);
        if size_acc + s_size < population {
            sizes.insert(s_repr, s_size);
            size_acc += s_size;
        } else {
            sizes.insert(s_repr, population - size_acc);
            break;
        }
    }
    sizes
}

fn main() {
    // start: rng, empty population
    // iter: evaluate population, bind fit,

    let mut rng = rng();
    let mut inno_head = 0;

    let mut population = population_init(2, 1, POPULATION, &mut rng);
    let mut gen_idx = 0;
    let game = GameXOR::new();
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

        let species_pop = specie_populations(&species, POPULATION, 0.4);
        if gen_idx == GENERATIONS {
            break species;
        }

        if gen_idx % 100 == 0{
            for (idx, s) in species.iter().filter(|s| !s.is_empty()).enumerate() {
                println!("champ {gen_idx}.{idx}: {}", s.1[0].1)
            }
        }

        let mut innogen = InnoGen::new(inno_head);
        population = species
            .iter()
            .flat_map(|specie| {
                specie
                    .reproduce(
                        *species_pop.get(&specie.0).unwrap_or(&0),
                        &mut innogen,
                        &mut rng,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        inno_head = innogen.head;
        gen_idx += 1;
    };

    println!("{gen_idx}");
    for (idx, s) in pop_evaluated.iter().filter(|s| !s.is_empty()).enumerate() {
        println!("champ {idx}: {}", s.1[0].1)
    }
}
