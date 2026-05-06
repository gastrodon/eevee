//! Traits and helpers for serializing and deserializing NeuroEvolution components.
//!
//! Format-specific implementations live in submodules (e.g. [`json`]).

#[cfg(feature = "serialize-json")]
pub mod json;

use core::error::Error;
use std::{
    fs::{self, read_dir},
    iter::empty,
    path::Path,
};

pub use crate::genome::NodeKind;

use crate::{
    population::{speciate, SpecieGroup},
    Connection, Genome, Specie,
};

/// Trait for types that can be serialized to/from a string and a file.
///
/// Implementations provide `to_str` and `from_str` for a specific format, and a
/// `SERIALIZER_ID` that is written as a header line in files. The default `to_file` /
/// `from_file` methods handle the header automatically and reject files that were
/// written by a different serializer.
pub trait SerializeFile: Sized {
    /// A short freeform identifier for this format, e.g. `"json-1"`.
    /// Written into every file as `eevee-serializer: <SERIALIZER_ID>`.
    const SERIALIZER_ID: &'static str;

    fn to_str(&self) -> Result<String, Box<dyn Error>>;

    #[allow(clippy::should_implement_trait)]
    fn from_str(s: &str) -> Result<Self, Box<dyn Error>>;

    fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let content = format!(
            "eevee-serializer: {}\n{}",
            Self::SERIALIZER_ID,
            self.to_str()?
        );
        fs::write(path, content)?;
        Ok(())
    }

    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let raw = fs::read_to_string(path)?;
        let body = strip_header(&raw, Self::SERIALIZER_ID)?;
        Self::from_str(body)
    }
}

fn strip_header<'a>(s: &'a str, expected_id: &str) -> Result<&'a str, Box<dyn Error>> {
    let expected = format!("eevee-serializer: {expected_id}");
    match s.split_once('\n') {
        Some((header, body)) if header == expected => Ok(body),
        Some((header, _)) => Err(format!("unexpected serializer header: {header:?}").into()),
        None => Err("missing serializer header".into()),
    }
}


/// Save a population of [Genome]s to individual files inside of a directory at `path`.
pub fn population_to_files<P: AsRef<Path>, C: Connection, G: Genome<C> + SerializeFile>(
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
pub fn population_from_files<P: AsRef<Path>, C: Connection, G: Genome<C> + SerializeFile>(
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

/// Load a single [Genome] from a file and clone it `population` times. Useful for resuming
/// training from a single champion, or inspecting a particular genome.
pub fn population_from_genome<
    P: AsRef<Path>,
    C: Connection,
    G: Genome<C> + SerializeFile + Clone,
>(
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
