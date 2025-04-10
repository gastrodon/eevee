use crate::{
    genome::{Connection, Genome, InnoGen},
    population::SpecieRepr,
    Specie,
};
use core::{error::Error, f64};
use rand::RngCore;
use std::collections::HashMap;

fn reproduce_crossover<C: Connection, G: Genome<C>>(
    genomes: &[(G, f64)],
    size: usize,
    rng: &mut impl RngCore,
    innogen: &mut InnoGen,
) -> Result<Vec<G>, Box<dyn Error>> {
    if size == 0 {
        return Ok(vec![]);
    }

    if genomes.len() < 2 {
        return Err(format!(
            "too few members to crossover (wanted to produce {size} from {}",
            genomes.len()
        )
        .into());
    }

    let pairs = {
        let mut pairs = genomes
            .iter()
            .enumerate()
            .flat_map(|(l_idx, (l, l_fit))| {
                genomes
                    .iter()
                    .enumerate()
                    .filter_map(move |(r_idx, (r, r_fit))| {
                        if l_fit > r_fit || (l_fit == r_fit && l_idx > r_idx) {
                            Some(((l, l_fit), (r, r_fit)))
                        } else {
                            None
                        }
                    })
            })
            .collect::<Vec<_>>();
        pairs.sort_by(|l, r| {
            let r = r.0 .1 + r.1 .1;
            let l = l.0 .1 + l.1 .1;
            (r).partial_cmp(&l)
                .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
        });
        pairs
    };

    pairs
        .into_iter()
        .cycle()
        .take(size)
        .map(|((l, _), (r, _))| {
            let mut child = l.reproduce_with(r, std::cmp::Ordering::Greater, rng);
            child.mutate(rng, innogen);
            Ok(child)
        })
        .collect()
}

fn reproduce_copy<C: Connection, G: Genome<C>>(
    genomes: &[(G, f64)],
    size: usize,
    rng: &mut impl RngCore,
    innogen: &mut InnoGen,
) -> Result<Vec<G>, Box<dyn Error>> {
    if size == 0 {
        return Ok(vec![]);
    }

    if genomes.is_empty() {
        return Err(format!(
            "too few members to copy (wanted to produce {size} from {}",
            genomes.len()
        )
        .into());
    }

    let mut top = genomes.iter().collect::<Vec<_>>();
    top.sort_by(|(_, l), (_, r)| {
        r.partial_cmp(l)
            .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
    });
    top.into_iter()
        .cycle()
        .take(size)
        .map(|(genome, _)| {
            let mut child = genome.clone();
            child.mutate(rng, innogen);
            Ok(child)
        })
        .collect()
}

pub fn reproduce<C: Connection, G: Genome<C>>(
    genomes: Vec<(G, f64)>,
    size: usize,
    innogen: &mut InnoGen,
    rng: &mut impl RngCore,
) -> Result<Vec<G>, Box<dyn Error>> {
    if size == 0 {
        return Ok(vec![]);
    }

    if genomes.is_empty() {
        return Err(format!(
            "too few members to reproduce (wanted to produce {size} from {}",
            genomes.len()
        )
        .into());
    }

    let mut pop: Vec<G> = Vec::with_capacity(size);
    pop.push(
        genomes
            .iter()
            .max_by(|(_, l), (_, r)| {
                l.partial_cmp(r)
                    .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
            })
            .unwrap()
            .0
            .clone(),
    );

    if size == 1 {
        return Ok(pop);
    }

    let size = size - 1;
    let size_copy = size / 4;
    let size_copy = if size_copy == 0 || genomes.len() == 1 {
        size
    } else {
        size_copy
    };

    // TODO reproduce_crossover and reproduce_copy can potentially be made faster
    // if they're handed a slice to write into intead of returning a vec that we then need to copy
    reproduce_copy(&genomes, size_copy, rng, innogen)?
        .into_iter()
        .for_each(|genome| pop.push(genome));

    let size_crossover = size - size_copy;
    reproduce_crossover(&genomes, size_crossover, rng, innogen)?
        .into_iter()
        .for_each(|genome| pop.push(genome));

    Ok(pop)
}

/// allocate a target population for every specie in an existing population
fn population_alloc<'a, C: Connection + 'a, G: Genome<C> + 'a>(
    species: impl Iterator<Item = &'a Specie<C, G>>,
    population: usize,
) -> HashMap<SpecieRepr<C>, usize> {
    let species_fitted = species
        .map(|s| (s.repr.clone(), s.fit_adjusted()))
        .collect::<Vec<_>>();

    let fit_total = species_fitted.iter().fold(0., |acc, (_, n)| acc + n);
    let population_f = population as f64;
    species_fitted
        .into_iter()
        .map(|(specie_repr, fit_adjusted)| {
            (
                specie_repr,
                f64::round(population_f * fit_adjusted / fit_total) as usize,
            )
        })
        .collect()
}

fn population_allocated<
    'a,
    C: Connection + 'a,
    G: Genome<C> + 'a,
    T: Iterator<Item = &'a (Specie<C, G>, f64)>,
>(
    species: T,
    population: usize,
) -> impl Iterator<Item = (Vec<(G, f64)>, usize)> {
    let viable = species
        .filter_map(|(specie, min_fitness)| {
            let viable = specie
                .members
                .iter()
                .filter(|&pair| (&pair.1 >= min_fitness))
                .cloned()
                .collect::<Vec<_>>();

            // (!viable.is_empty()).then_some((&specie.repr, viable));
            (!viable.is_empty()).then(|| Specie {
                repr: specie.repr.clone(),
                members: viable,
            })
        })
        .collect::<Vec<_>>();

    let alloc = population_alloc(viable.iter(), population);

    viable
        .into_iter()
        .filter_map(move |specie| alloc.get(&specie.repr).map(|pop| (specie.members, *pop)))
}

/// Reproduce a group of species, allocating their populations based on their specie fitness
/// relative to eachother. Enforces a min_fitness threshold for every specie member, and allows
/// low-fitness species to naturally die off.
pub fn population_reproduce<C: Connection, G: Genome<C>>(
    species: &[(Specie<C, G>, f64)],
    population: usize,
    inno_head: usize,
    rng: &mut impl RngCore,
) -> (Vec<G>, usize) {
    // let species = population_viable(species.into_iter());
    // let species_pop = population_alloc(species, population);
    let mut innogen = InnoGen::new(inno_head);
    (
        population_allocated(species.iter(), population)
            .flat_map(|(members, pop)| reproduce(members, pop, &mut innogen, rng).unwrap())
            .collect::<Vec<_>>(),
        innogen.head,
    )
}

#[cfg(test)]
mod test {
    use crate::{
        genome::{Recurrent, WConnection},
        population::population_init,
        random::default_rng,
        test_t,
    };

    use super::*;

    #[test]
    fn test_inno_gen() {
        let mut inno = InnoGen::new(0);
        assert_eq!(inno.head, 0);
        assert_eq!(inno.path((0, 1)), 0);
        assert_eq!(inno.path((1, 2)), 1);
        assert_eq!(inno.path((0, 1)), 0);
        assert_eq!(inno.head, 2);

        let mut inno2 = InnoGen::new(inno.head);
        assert_eq!(inno2.path((1, 0)), 2);
        assert_eq!(inno2.path((0, 1)), 3);
    }

    type BasicGenomeCtrnn = Recurrent<WConnection>;

    test_t!(specie_reproduce[T: BasicGenomeCtrnn]() {
        let mut rng = default_rng();
        let count = 40;
        let (species, inno_head) = population_init::<WConnection, T>(2, 2, count);

        for specie in species {
            for i in [0, 1, count, count * 10] {
                assert_eq!(
                    i,
                    reproduce(
                        specie.members.clone(),
                        i,
                        &mut InnoGen::new(inno_head),
                        &mut rng
                    )
                    .unwrap()
                    .len()
                );
            }
        }
    });
}
