use std::collections::HashMap;

use crate::{
    network::Network,
    specie::{population_reproduce, speciate, Specie, SpecieRepr},
};
use rand::rng;

pub enum EvolutionTarget {
    Fitness(f64),
    Generation(usize),
}

impl EvolutionTarget {
    fn satisfied(&self, species: &[Specie], generation: usize) -> bool {
        if let Self::Fitness(t) = self {
            species
                .iter()
                .any(|f| f.members.first().is_some_and(|(_, f)| f >= t))
        } else if let Self::Generation(t) = self {
            t <= &generation
        } else {
            false
        }
    }
}

pub trait Scenario {
    fn io() -> (usize, usize);
    fn eval<F: Fn(f64) -> f64>(&self, network: &mut impl Network, σ: F) -> f64;

    fn evolve(
        &self,
        target: EvolutionTarget,
        init: impl FnOnce((usize, usize)) -> (Vec<Specie>, usize),
        population_lim: usize,
        σ: impl Fn(f64) -> f64,
    ) -> (Vec<Specie>, usize) {
        let (mut pop_flat, mut inno_head) = {
            let (species, inno_head) = init(Self::io());
            (
                species
                    .iter()
                    .flat_map(|Specie { members, .. }| {
                        members.iter().map(|(genome, _)| genome.clone())
                    })
                    .collect::<Vec<_>>(),
                inno_head,
            )
        };

        let mut scores: HashMap<SpecieRepr, _> = HashMap::new();

        let mut rng = rng();
        let mut gen_idx = 0;
        loop {
            let species = speciate(pop_flat.into_iter().map(|genome| {
                let fitness = self.eval(&mut genome.network(), &σ);
                (genome, fitness)
            }));

            if target.satisfied(&species, gen_idx) {
                break (species, inno_head);
            };

            let scores_prev = scores;
            scores = species
                .iter()
                .filter_map(|Specie { repr, members }| {
                    members
                        .iter()
                        .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
                        .map(|(_, max)| (repr.clone(), *max))
                })
                .collect();

            let p_scored = species
                .into_iter()
                .map(|s| {
                    let min_fit = *scores_prev.get(&s.repr).unwrap_or(&0.);
                    (s, min_fit)
                })
                .collect::<Vec<_>>();

            (pop_flat, inno_head) =
                population_reproduce(&p_scored, population_lim, inno_head, &mut rng);

            gen_idx += 1
        }
    }
}
