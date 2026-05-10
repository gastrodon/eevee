//! Traits related to evaluation, fitting, and evolution of genomes for specific tasks.

use crate::{
    genome::Genome,
    population::{speciate, Specie, SpecieRepr},
    reproduce::population_reproduce,
    Connection,
};
use core::{f64, ops::ControlFlow};
use rand::RngCore;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;

/// Tunable parameters governing speciation, stagnation, and reproduction.
pub struct EvolutionConfig {
    /// Compatibility distance below which two genomes are placed in the same species.
    pub specie_threshold: f64,
    /// Consecutive generations without fitness improvement before a species is penalised.
    pub no_improvement_truncate: usize,
    /// Members retained when a stagnant species is truncated (kill if at or below this).
    pub no_improvement_floor: usize,
    /// Crossover/copy split denominator: 1/n offspring are copy-mutants, rest are crossover.
    pub copy_denom: usize,
    /// Fitness multiplier for a brand-new species (age 0); decays linearly to 1.0.
    pub specie_youth_fac: f64,
    /// Age at which the youth multiplier reaches 1.0.
    pub specie_youth_dropoff: usize,
    /// Minimum population slots allocated to any surviving species.
    pub specie_min_pop: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            specie_threshold: 4.0,
            no_improvement_truncate: 10,
            no_improvement_floor: 2,
            copy_denom: 4,
            specie_youth_fac: 2.0,
            specie_youth_dropoff: 10,
            specie_min_pop: 2,
        }
    }
}

/// Stats passed to a hook fn
pub struct Stats<'a, C: Connection, G: Genome<C>> {
    pub generation: usize,
    pub species: &'a [Specie<C, G>],
}

impl<C: Connection, G: Genome<C>> Stats<'_, C, G> {
    pub fn any_fitter_than(&self, target: f64) -> bool {
        self.species
            .iter()
            .any(|Specie { members, .. }| members.iter().any(|(_, fitness)| *fitness > target))
    }

    pub fn fittest(&self) -> Option<&(G, f64)> {
        self.species
            .iter()
            .flat_map(|Specie { members, .. }| members.iter())
            .max_by(|(_, l), (_, r)| {
                l.partial_cmp(r)
                    .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
            })
    }
}

pub type Hook<C, G> = Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>>;

/// Functions that hook into the evolution process, allowing observation and mutation.
/// Each hook is called each generation with a [Stats] exposing the current population and
/// generation number. Hooks are called the order that they're provided in `new`.
///
/// Hooks can halt evolution, causing `evolve` to return, by returning a ControlFlow::Break
pub struct EvolutionHooks<C: Connection, G: Genome<C>> {
    hooks: Vec<Hook<C, G>>,
}

impl<C: Connection, G: Genome<C>> EvolutionHooks<C, G> {
    pub fn new(hooks: Vec<Hook<C, G>>) -> Self {
        Self { hooks }
    }

    fn fire(&self, mut stats: Stats<C, G>) -> ControlFlow<()> {
        for hook in self.hooks.iter() {
            if hook(&mut stats).is_break() {
                return ControlFlow::Break(());
            }
        }

        ControlFlow::Continue(())
    }
}

/// Scenario describes the setting in which evolution takes place. For any genome kind,
/// (eval)[Scenario::eval] should be implemented such that it evaluates the genome ( or a
/// network that it produces ) with some fitness. Greater fitnesses will be optimized for
pub trait Scenario<C: Connection, G: Genome<C>, A: Fn(f64) -> f64> {
    fn io(&self) -> (usize, usize);
    fn eval(&self, genome: &G, σ: &A) -> f64;
}

/// Given a well-defined evolution scenario, evolve is the entrypoint into actually... evolving.
/// It will manage evaluation, speciation, reproduction, and mutation of a pool of genomes
/// about ( but not necessarily exactly ) `population` large. Each specie is allocated some size
/// in terms of `population`.
///
/// If compiled with `--features parallel`, evaluation will run in a thread-pool of one thread
/// per cpu on the host. This in turn requires our arguments ( excluding init, which is called
/// exactly once ) to implement [Sync]
pub fn evolve<
    C: Connection,
    #[cfg(not(feature = "parallel"))] G: Genome<C>,
    #[cfg(feature = "parallel")] G: Genome<C> + Send,
    I: FnOnce((usize, usize)) -> (Vec<Specie<C, G>>, usize),
    #[cfg(not(feature = "parallel"))] A: Fn(f64) -> f64,
    #[cfg(feature = "parallel")] A: Fn(f64) -> f64 + Sync,
    #[cfg(not(feature = "parallel"))] S: Scenario<C, G, A>,
    #[cfg(feature = "parallel")] S: Scenario<C, G, A> + Sync,
>(
    scenario: S,
    init: I,
    σ: A,
    mut rng: impl RngCore,
    hooks: EvolutionHooks<C, G>,
    config: EvolutionConfig,
) -> (Vec<Specie<C, G>>, usize) {
    let (mut pop_flat, mut inno_head) = {
        let (species, inno_head) = init(scenario.io());
        (
            species
                .iter()
                .flat_map(|Specie { members, .. }| members.iter().map(|(genome, _)| genome.clone()))
                .collect::<Vec<_>>(),
            inno_head,
        )
    };

    let population_lim = pop_flat.len();

    // scores: repr → (best_fitness, gen_last_improved, gen_born)
    let mut scores: HashMap<SpecieRepr<C>, (f64, usize, usize)> = HashMap::new();
    let mut gen_idx = 0;
    loop {
        let species = {
            #[cfg(not(feature = "parallel"))]
            let genomes = pop_flat.into_iter().map(|genome| {
                let fitness = scenario.eval(&genome, &σ);
                (genome, fitness)
            });
            #[cfg(feature = "parallel")]
            let genomes = pop_flat
                .into_par_iter()
                .map(|genome| {
                    let fitness = scenario.eval(&genome, &σ);
                    (genome, fitness)
                })
                .collect::<Vec<_>>()
                .into_iter();
            let reprs = scores
                .iter()
                .map(|(repr, (_, _, born))| (repr.clone(), *born));

            #[cfg(not(feature = "smol_bench"))]
            let species = speciate(genomes, reprs, gen_idx, config.specie_threshold);
            #[cfg(feature = "smol_bench")]
            let species = speciate(
                genomes.collect::<Vec<_>>().into_iter(),
                reprs.collect::<Vec<_>>().into_iter(),
                gen_idx,
                config.specie_threshold,
            );
            species
        };

        if hooks
            .fire(Stats {
                generation: gen_idx,
                species: &species,
            })
            .is_break()
        {
            break (species, inno_head);
        }

        let scores_prev = scores;
        scores = species
            .iter()
            .filter_map(
                |Specie {
                     repr,
                     members,
                     born,
                     ..
                 }| {
                    let gen_max = members.iter().max_by(|(_, l), (_, r)| {
                        l.partial_cmp(r)
                            .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
                    });
                    let past_max = scores_prev.get(repr);

                    match (gen_max, past_max) {
                        (Some((_, gen_max)), Some((past_max, past_idx, past_born))) => {
                            if gen_max > past_max {
                                Some((repr.clone(), (*gen_max, gen_idx, *past_born)))
                            } else {
                                Some((repr.clone(), (*past_max, *past_idx, *past_born)))
                            }
                        }
                        (Some((_, gen_max)), None) => {
                            Some((repr.clone(), (*gen_max, gen_idx, *born)))
                        }
                        // Species went empty but had history — keep its repr in scores so
                        // speciation next generation can still match genomes to it. Without
                        // this, the niche is permanently lost the moment a species empties.
                        (None, Some(past)) => Some((repr.clone(), *past)),
                        (None, None) => None,
                    }
                },
            )
            .collect();

        let p_truncated = species
            .into_iter()
            .filter_map(|s| {
                let (_, gen_achieved, _) =
                    *scores.get(&s.repr).unwrap_or(&(f64::MIN, gen_idx, gen_idx));
                let stagnant = gen_achieved + config.no_improvement_truncate <= gen_idx;

                if stagnant && s.members.len() > config.no_improvement_floor {
                    Some(Specie {
                        repr: s.repr,
                        born: s.born,
                        members: {
                            let mut trunc = s.members;
                            trunc.sort_by(|(_, l), (_, r)| {
                                r.partial_cmp(l)
                                    .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
                            });
                            trunc[..config.no_improvement_floor].to_vec()
                        },
                    })
                } else if stagnant {
                    None // already minimal and still stagnant — kill
                } else {
                    Some(s)
                }
            })
            .collect::<Vec<_>>();

        (pop_flat, inno_head) = population_reproduce(
            &p_truncated,
            population_lim,
            gen_idx,
            inno_head,
            &mut rng,
            &config,
        );
        debug_assert!(!pop_flat.is_empty(), "nobody past {gen_idx}");
        gen_idx += 1
    }
}
