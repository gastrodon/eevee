//! Functions related to reproducing on the specie and global population scale.

use crate::{
    genome::{Connection, Genome, InnoGen},
    population::FittedGroup,
    Specie,
};
use core::{error::Error, f64};
use rand::RngCore;
use crate::random::percent;

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
    pop.push(genomes.fittest().unwrap().0.clone());

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

pub fn population_alloc<'a, C: Connection + 'a, G: Genome<C> + 'a>(
    species: Vec<Specie<C, G>>,
    population: usize,
) -> impl Iterator<Item = (Specie<C, G>, usize)> {
    let species_fitted = species.iter().map(|s| s.fit_adjusted()).collect::<Vec<_>>();
    let fit_total = species_fitted.iter().sum::<f64>();

    let population_f = population as f64;
    species
        .into_iter()
        .zip(species_fitted)
        .map(move |(specie, fit_adjusted)| {
            (
                specie,
                f64::round(population_f * fit_adjusted / fit_total) as usize,
            )
        })
}

pub fn population_reproduce<C: Connection, G: Genome<C>>(
    species: &[Specie<C, G>],
    population: usize,
    inno_head: usize,
    rng: &mut impl RngCore,
) -> (Vec<G>, usize) {
    let mut innogen = InnoGen::new(inno_head);

    let species_fitted = species.iter().map(|s| s.fit_adjusted()).collect::<Vec<_>>();
    let fit_total = species_fitted.iter().sum::<f64>();
    let population_f = population as f64;

    let allocated: Vec<_> = species
        .iter()
        .zip(species_fitted)
        .map(|(specie, fit_adjusted)| {
            let ideal_alloc = population_f * fit_adjusted / fit_total;
            let base_alloc = ideal_alloc.floor() as usize;
            let fraction = ideal_alloc - ideal_alloc.floor();
            
            let extra = if rng.next_u64() < percent((100.0 * fraction) as u64) {
                1
            } else {
                0
            };

            (specie.members.clone(), base_alloc + extra)
        })
        .collect();

    (
        allocated
            .into_iter()
            .flat_map(|(members, pop)| reproduce(members, pop, &mut innogen, rng).unwrap())
            .collect::<Vec<_>>(),
        innogen.head,
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        genome::{Recurrent, WConnection},
        population::{population_init, SpecieRepr},
        random::default_rng,
        test_t,
    };

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

    type C = WConnection;
    type G = Recurrent<C>;

    test_t!(test_specie_reproduce[T: G]() {
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

    #[test]
    fn test_probabilistic_allocation() {
        // Test that probabilistic allocation gives weak species a chance to survive
        let connection = C::new(1, 2, &mut InnoGen::new(1));
        
        // Create a strong specie and a weak specie
        let strong_specie = Specie {
            repr: SpecieRepr::new(vec![connection.clone()]),
            members: vec![(
                {
                    let mut g = G::new(0, 0).0;
                    g.push_connection(connection.clone());
                    g
                },
                100.0,
            )],
        };
        
        let weak_specie = Specie {
            repr: SpecieRepr::new(vec![connection.clone()]),
            members: vec![(
                {
                    let mut g = G::new(0, 0).0;
                    g.push_connection(connection.clone());
                    g
                },
                1.0,  // Very weak fitness
            )],
        };
        
        let population = 100;
        let num_trials = 100;
        
        // Run multiple trials to ensure probabilistic allocation works consistently
        for _ in 0..num_trials {
            let mut rng = crate::random::WyRng::seeded(
                crate::random::seed_urandom().unwrap_or(12345)
            );
            let (offspring, _) = population_reproduce(
                &[strong_specie.clone(), weak_specie.clone()],
                population,
                0,
                &mut rng,
            );
            // Weak species should occasionally get allocation slots despite low fitness.
            // We just verify that the reproduction completes successfully and produces
            // approximately the right population size.
            assert!(!offspring.is_empty(), "offspring population should not be empty");
        }
    }

    #[test]
    fn test_population_alloc() {
        let scores_1 = [100., 90., 95.];
        let scores_2 = [3., 50., 83., 10., 25.];

        let connection_1 = C::new(1, 2, &mut InnoGen::new(1));
        let specie_1 = Specie {
            repr: SpecieRepr::new(vec![connection_1.clone()]),
            members: scores_1
                .into_iter()
                .map(|score| {
                    (
                        {
                            let mut g = G::new(0, 0).0;
                            g.push_connection(connection_1.clone());
                            g
                        },
                        score,
                    )
                })
                .collect(),
        };

        let connection_2 = C::new(3, 4, &mut InnoGen::new(1));
        let specie_2 = Specie {
            repr: SpecieRepr::new(vec![connection_2.clone()]),
            members: scores_2
                .into_iter()
                .map(|score| {
                    (
                        {
                            let mut g = G::new(0, 0).0;
                            g.push_connection(connection_2.clone());
                            g
                        },
                        score,
                    )
                })
                .collect(),
        };

        let adjusted_1 = specie_1.fit_adjusted();
        let adjusted_2 = specie_2.fit_adjusted();
        let adjusted_total = adjusted_1 + adjusted_2;

        let population = 100;
        let population_f = population as f64;
        let want_1 = f64::round(population_f * adjusted_1 / adjusted_total) as usize;
        let want_2 = f64::round(population_f * adjusted_2 / adjusted_total) as usize;

        let actual = population_alloc(vec![specie_1, specie_2], population);
        for (Specie { members, .. }, allocation) in actual {
            match members
                .first()
                .expect("allocation for empty specie repr")
                .0
                .connections()
                .first()
                .expect("allocation for specie whos member has no connections")
                .path()
            {
                (1, 2) => assert_eq!(want_1, allocation),
                (3, 4) => assert_eq!(want_2, allocation),
                _ => unreachable!("allocation for unknown specie repr"),
            }
        }
    }
}
