//! JSON serialization helpers for populations of genomes.

use crate::{
    genome::{Connection, Genome},
    population::{speciate, Specie, SpecieGroup},
};
use core::error::Error;
use std::{fs::read_dir, iter::empty, path::Path};

/// Save a population of [Genome]s to individual files inside of a directory at `path`
pub fn population_to_files<P: AsRef<Path>, C: Connection, G: Genome<C>>(
    path: P,
    pop: &[Specie<C, G>],
) -> Result<(), Box<dyn Error>>
where
    G: serde::Serialize,
{
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
) -> Result<SpecieGroup<C, G>, Box<dyn Error>>
where
    G: for<'de> serde::Deserialize<'de>,
{
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
) -> Result<SpecieGroup<C, G>, Box<dyn Error>>
where
    G: for<'de> serde::Deserialize<'de>,
{
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
