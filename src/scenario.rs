use crate::{
    genome::Genome,
    network::Network,
    specie::{population_reproduce, speciate, Specie},
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
                .any(|f| f.1.first().is_some_and(|(_, f)| f >= t))
        } else if let Self::Generation(t) = self {
            t <= &generation
        } else {
            false
        }
    }
}

pub trait Scenario {
    fn io() -> (usize, usize);
    fn eval(&self, network: &mut impl Network) -> f64;

    fn evolve(
        &self,
        target: EvolutionTarget,
        init: impl FnOnce((usize, usize)) -> (Vec<Genome>, usize),
        population_lim: usize,
        σ: impl Fn(f64) -> f64,
        genome_top_p: f64,
        specie_top_p: f64,
    ) -> (Vec<(Genome, f64)>, usize) {
        let (mut pop_unspeciated, mut inno_head) = init(Self::io());

        let mut rng = rng();
        let mut gen_idx = 0;
        loop {
            let scored = pop_unspeciated
                .iter()
                .map(|genome| (genome, self.eval(&mut genome.network(&σ))))
                .collect::<Vec<_>>();

            let species = {
                let mut species = speciate(scored.into_iter());
                for s in species.iter_mut() {
                    s.shrink_top_p(genome_top_p);
                }
                species
            };

            if target.satisfied(&species, gen_idx) {
                break (
                    species.iter().flat_map(|s| s.cloned().1).collect(),
                    inno_head,
                );
            };

            (pop_unspeciated, inno_head) =
                population_reproduce(&species, population_lim, specie_top_p, inno_head, &mut rng);

            gen_idx += 1
        }
    }
}
