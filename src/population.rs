//! Functions and structs related to managing genomes at the specie and global population scale.

use crate::{
    crossover::delta,
    genome::{Connection, Genome},
};
use core::{
    f64,
    hash::{Hash, Hasher},
};
use std::hash::DefaultHasher;

/// The representative member of a particular specie. Is retained inter-generationally to better
/// track when a specie deviates
#[derive(Debug, Clone)]
pub struct SpecieRepr<C: Connection>(Vec<C>);

impl<C: Connection> SpecieRepr<C> {
    pub fn new(v: Vec<C>) -> Self {
        Self(v)
    }

    fn delta(&self, other: &[C]) -> f64 {
        delta(&self.0, other)
    }

    #[inline]
    fn cloned(&self) -> Vec<C> {
        self.0.to_vec()
    }
}

impl<C: Connection> SpecieRepr<C> {
    fn id(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

impl<C: Connection> Hash for SpecieRepr<C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<C: Connection> PartialEq for SpecieRepr<C> {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<C: Connection> Eq for SpecieRepr<C> {}

impl<C: Connection> AsRef<[C]> for SpecieRepr<C> {
    fn as_ref(&self) -> &[C] {
        &self.0
    }
}

pub trait FittedGroup<T> {
    fn fittest(&self) -> Option<&(T, f64)>;
    fn fit_adjusted(&self) -> f64;
}

impl<T> FittedGroup<T> for [(T, f64)] {
    fn fittest(&self) -> Option<&(T, f64)> {
        self.iter().max_by(|(_, l), (_, r)| {
            l.partial_cmp(r)
                .unwrap_or_else(|| panic!("could not compare specie member fitness {l} to {r}"))
        })
    }

    fn fit_adjusted(&self) -> f64 {
        let l = self.len() as f64;
        self.iter().fold(0., |acc, (_, fit)| acc + *fit / l)
    }
}

/// A collection of fitted [Genome]s who are closely related to the same [SpecieRepr]
#[derive(Debug, Clone)]
pub struct Specie<C: Connection, G: Genome<C>> {
    pub repr: SpecieRepr<C>,
    pub members: Vec<(G, f64)>,
    /// Generation in which this species was first created.
    pub born: usize,
}

impl<C: Connection, G: Genome<C>> Specie<C, G> {
    #[inline]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    #[inline]
    pub fn last(&self) -> Option<&(G, f64)> {
        self.members.last()
    }

    #[inline]
    pub fn cloned(&self) -> (Vec<C>, Vec<(G, f64)>) {
        (
            self.repr.cloned(),
            self.members.iter().map(|(g, s)| (g.clone(), *s)).collect(),
        )
    }
}

impl<C: Connection, G: Genome<C>> FittedGroup<G> for Specie<C, G> {
    fn fit_adjusted(&self) -> f64 {
        self.members.fit_adjusted()
    }

    fn fittest(&self) -> Option<&(G, f64)> {
        self.members.fittest()
    }
}

/// Compatibility distance threshold for speciation.
///
/// Two genomes are placed in the same species when `delta(a, b) < SPECIE_THRESHOLD`.
/// `delta` is (disjoint + excess genes) / max_genome_size + 0.4 * avg_weight_diff,
/// so it is roughly "fraction of genes that don't align, plus a weight penalty".
///
/// Practical intuition:
/// - 0.0 → only identical genomes share a species (useless)
/// - 0.5 → ~50% structural mismatch or large weight drift splits a species
/// - 1.5 → two completely non-overlapping equal-size genomes (delta≈2.0) still split;
///          a genome with one extra bisection (delta≈0.2) stays in the same species
const SPECIE_THRESHOLD: f64 = 1.5;

/// Partition an unordered collection of [Genome]s into species. An initial collection of empty
/// species is created from repr, and if some genome matches none of them, a new specie is
/// formed with them as the repr.
pub fn speciate<C: Connection, G: Genome<C>>(
    genomes: impl Iterator<Item = (G, f64)>,
    reprs: impl Iterator<Item = (SpecieRepr<C>, usize)>,
    generation: usize,
) -> Vec<Specie<C, G>> {
    let mut sp = Vec::from_iter(reprs.map(|(repr, born)| Specie {
        repr,
        members: Vec::new(),
        born,
    }));

    for (genome, fitness) in genomes {
        let best = sp
            .iter_mut()
            .filter_map(|s| {
                let d = s.repr.delta(genome.connections());
                if d < SPECIE_THRESHOLD { Some((d, s)) } else { None }
            })
            .min_by(|(dl, _), (dr, _)| {
                dl.partial_cmp(dr).unwrap_or(core::cmp::Ordering::Equal)
            });

        match best {
            Some((_, Specie { members, .. })) => members.push((genome, fitness)),
            None => {
                sp.push(Specie {
                    repr: SpecieRepr::new(genome.connections().to_vec()),
                    members: vec![(genome, fitness)],
                    born: generation,
                });
            }
        }
    }

    sp
}

pub type SpecieGroup<C, G> = (Vec<Specie<C, G>>, usize);

/// initial population of a single specie consisting of single connection genomes
/// while it's not necessarily recommended to do an initual mutation, it allows us to mutate a
/// bisection on any genome without the need to check for existing connections beforehand
pub fn population_init<C: Connection, G: Genome<C>>(
    sensory: usize,
    action: usize,
    population: usize,
) -> SpecieGroup<C, G> {
    let (genome, inno_head) = G::new(sensory, action);
    (
        vec![Specie {
            repr: SpecieRepr::new(genome.connections().to_vec()),
            members: vec![(genome, f64::MIN); population],
            born: 0,
        }],
        inno_head,
    )
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        genome::{Recurrent, WConnection},
        test_t,
    };

    type BasicGenomeCtrnn = Recurrent<WConnection>;

    test_t!(test_population_init[T: BasicGenomeCtrnn]() {
        let count = 40;
        let (species, inno_head) = population_init::<WConnection, T>(2, 2, count);
        assert_eq!(
            count,
            species
                .iter()
                .fold(0, |acc, Specie { members, .. }| acc + members.len())
        );
        assert!(species
            .iter()
            .flat_map(|specie| specie.members.iter().flat_map(|(member, _)| member
                .connections()
                .iter()
                .map(|connection| connection.inno())))
            .all(|inno| inno < inno_head));
        for specie in species.iter() {
            assert_ne!(0, specie.len());
        }
        for (_, fit) in species.iter().flat_map(|Specie { members, .. }| members) {
            assert_eq!(f64::MIN, *fit);
        }
    });
}
