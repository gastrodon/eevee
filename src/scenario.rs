use crate::{
    genome::Genome,
    specie::{population_reproduce, speciate, Specie, SpecieRepr},
    Connection, Node,
};
use core::{f64, ops::ControlFlow};
use rand::RngCore;
#[cfg(feature = "parallel")]
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPoolBuilder,
};
use std::{collections::HashMap, marker::PhantomData};

const NO_IMPROVEMENT_TRUNCATE: usize = 10;

pub struct Stats<'a, N: Node, C: Connection, G: Genome<N, C>> {
    pub generation: usize,
    pub species: &'a [Specie<N, C, G>],
}

impl<N: Node, C: Connection, G: Genome<N, C>> Stats<'_, N, C, G> {
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

pub type Hook<N, C, G> = Box<dyn Fn(&mut Stats<'_, N, C, G>) -> ControlFlow<()>>;

pub struct EvolutionHooks<N: Node, C: Connection, G: Genome<N, C>> {
    hooks: Vec<Hook<N, C, G>>,
}

impl<N: Node, C: Connection, G: Genome<N, C>> EvolutionHooks<N, C, G> {
    pub fn new(hooks: Vec<Hook<N, C, G>>) -> Self {
        Self { hooks }
    }

    fn fire(&self, mut stats: Stats<N, C, G>) -> ControlFlow<()> {
        for hook in self.hooks.iter() {
            if hook(&mut stats).is_break() {
                return ControlFlow::Break(());
            }
        }

        ControlFlow::Continue(())
    }
}

pub trait Scenario<N: Node, C: Connection, G: Genome<N, C>, A: Fn(f64) -> f64> {
    fn io(&self) -> (usize, usize);
    fn eval(&self, genome: &G, σ: &A) -> f64;
}

pub fn evolve<
    N: Node,
    C: Connection,
    #[cfg(not(feature = "parallel"))] G: Genome<N, C>,
    #[cfg(feature = "parallel")] G: Genome<N, C> + Send,
    I: FnOnce((usize, usize)) -> (Vec<Specie<N, C, G>>, usize),
    #[cfg(not(feature = "parallel"))] A: Fn(f64) -> f64,
    #[cfg(feature = "parallel")] A: Fn(f64) -> f64 + Sync,
    #[cfg(not(feature = "parallel"))] S: Scenario<N, C, G, A>,
    #[cfg(feature = "parallel")] S: Scenario<N, C, G, A> + Sync,
>(
    scenario: S,
    init: I,
    σ: A,
    mut rng: impl RngCore,
    hooks: EvolutionHooks<N, C, G>,
) -> (Vec<Specie<N, C, G>>, usize) {
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

    #[cfg(feature = "parallel")]
    let thread_pool = ThreadPoolBuilder::new().build().unwrap();
    let population_lim = pop_flat.len();

    let mut scores: HashMap<SpecieRepr<C>, _> = HashMap::new();
    let mut gen_idx = 0;
    loop {
        let species = {
            #[cfg(not(feature = "parallel"))]
            let genomes = pop_flat.into_iter().map(|genome| {
                let fitness = scenario.eval(&genome, &σ);
                (genome, fitness)
            });
            #[cfg(feature = "parallel")]
            let genomes = thread_pool.install(|| {
                pop_flat
                    .into_par_iter()
                    .map(|genome| {
                        let fitness = scenario.eval(&genome, &σ);
                        (genome, fitness)
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            });
            let reprs = scores.keys().cloned();

            #[cfg(not(feature = "smol_bench"))]
            let species = speciate(genomes, reprs);
            #[cfg(feature = "smol_bench")]
            let species = speciate(
                genomes.collect::<Vec<_>>().into_iter(),
                reprs.collect::<Vec<_>>().into_iter(),
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
            .filter_map(|Specie { repr, members, .. }| {
                let gen_max = members.iter().max_by(|(_, l), (_, r)| {
                    l.partial_cmp(r)
                        .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
                });
                let past_max = scores_prev.get(repr);

                match (gen_max, past_max) {
                    (Some((_, gen_max)), Some((past_max, past_idx))) => {
                        if gen_max > past_max {
                            Some((repr.clone(), (*gen_max, gen_idx)))
                        } else {
                            Some((repr.clone(), (*past_max, *past_idx)))
                        }
                    }
                    (Some((_, gen_max)), None) => Some((repr.clone(), (*gen_max, gen_idx))),
                    (None, _) => None,
                }
            })
            .collect();

        let p_scored = species
            .into_iter()
            .map(|s| {
                let (min_fit, gen_achieved) =
                    *scores_prev.get(&s.repr).unwrap_or(&(f64::MIN, gen_idx));

                if gen_achieved + NO_IMPROVEMENT_TRUNCATE <= gen_idx && s.members.len() > 2 {
                    (
                        Specie {
                            repr: s.repr,
                            members: {
                                let mut trunc = s.members;
                                trunc.sort_by(|(_, l), (_, r)| {
                                    r.partial_cmp(l)
                                        .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
                                });
                                trunc[..2].to_vec()
                            },
                            phantom: PhantomData,
                        },
                        f64::MIN,
                    )
                } else {
                    (s, min_fit)
                }
            })
            .collect::<Vec<_>>();

        (pop_flat, inno_head) =
            population_reproduce(&p_scored, population_lim, inno_head, &mut rng);
        debug_assert!(!pop_flat.is_empty(), "nobody past {gen_idx}");
        gen_idx += 1
    }
}
