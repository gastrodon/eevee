//! Functions and structs related to managing genomes at the specie and global population scale.

use crate::{
    crossover::delta,
    genome::{Connection, Genome},
};
use core::{
    error::Error,
    f64,
    hash::{Hash, Hasher},
};
use std::{fs::read_dir, hash::DefaultHasher, iter::empty, path::Path};

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

/// A collection of fitted [Genome]s who are closely related to the same [SpecieRepr]
#[derive(Debug)]
pub struct Specie<C: Connection, G: Genome<C>> {
    pub repr: SpecieRepr<C>,
    pub members: Vec<(G, f64)>,
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

    pub fn fit_adjusted(&self) -> f64 {
        let l = self.len() as f64;
        self.members.iter().fold(0., |acc, (_, fit)| acc + *fit / l)
    }
}

const SPECIE_THRESHOLD: f64 = 4.;

/// Partition an unordered collection of [Genome]s into species. An initial collection of empty
/// species is created from repr, and if some genome matches none of them, a new specie is
/// formed with them as the repr.
pub fn speciate<C: Connection, G: Genome<C>>(
    genomes: impl Iterator<Item = (G, f64)>,
    reprs: impl Iterator<Item = SpecieRepr<C>>,
) -> Vec<Specie<C, G>> {
    let mut sp = Vec::from_iter(reprs.map(|repr| Specie {
        repr,
        members: Vec::new(),
    }));

    for (genome, fitness) in genomes {
        match sp
            .iter_mut()
            .find(|Specie { repr, .. }| repr.delta(genome.connections()) < SPECIE_THRESHOLD)
        {
            Some(Specie { members, .. }) => members.push((genome, fitness)),
            None => {
                sp.push(Specie {
                    repr: SpecieRepr::new(genome.connections().to_vec()),
                    members: vec![(genome, fitness)],
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
        }],
        inno_head,
    )
}

/// Save a population of [Genome]s to individual files inside of a directory at `path`
pub fn population_to_files<P: AsRef<Path>, C: Connection, G: Genome<C>>(
    path: P,
    pop: &[Specie<C, G>],
) -> Result<(), Box<dyn Error>> {
    for (idx, (member, _)) in pop
        .iter()
        .flat_map(|specie| specie.members.iter())
        .enumerate()
    {
        member.to_file(path.as_ref().join(format!("{idx}.json")))?;
    }

    Ok(())
}

/// Load a population of [Genome]s from individual files inside of a directory at `path`. Assumes
/// that every file in `path` is a valid descriptor, and will parse it.
pub fn population_from_files<P: AsRef<Path>, C: Connection, G: Genome<C>>(
    path: P,
) -> Result<SpecieGroup<C, G>, Box<dyn Error>> {
    let pop_flat = read_dir(path)?
        .map(|fp| Ok::<_, Box<dyn Error>>((G::from_file(fp?.path())?, f64::MIN)))
        .collect::<Result<Vec<_>, _>>()?;

    if pop_flat.is_empty() {
        return Err("no genomes".into());
    }

    let inno_head = pop_flat
        .iter()
        .flat_map(|(g, _)| g.connections().iter().map(|c| c.inno()))
        .max()
        .unwrap_or(0);

    Ok((speciate(pop_flat.into_iter(), empty()), inno_head))
}

/// Load a single [Genome] from a single file, and clone it `population` times. Useful for
/// resuming training from a single champion, or inspecting a particular genome.
pub fn population_from_genome<P: AsRef<Path>, C: Connection, G: Genome<C>>(
    path: P,
    population: usize,
) -> Result<SpecieGroup<C, G>, Box<dyn Error>> {
    let muse = G::from_file(path)?;
    let inno_head = muse
        .connections()
        .iter()
        .map(|c| c.inno())
        .max()
        .unwrap_or(0);

    Ok((
        speciate(vec![(muse, f64::MIN); population].into_iter(), empty()),
        inno_head,
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        genome::{Recurrent, WConnection},
        test_t,
    };

    type BasicGenomeCtrnn = Recurrent<WConnection>;

    test_t!(population_init[T: BasicGenomeCtrnn]() {
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
        for (genome, fit) in species.iter().flat_map(|Specie { members, .. }| members) {
            assert_eq!(0, genome.connections().len());
            assert_eq!(f64::MIN, *fit);
        }
    });
}
