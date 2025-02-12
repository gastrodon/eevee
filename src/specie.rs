use crate::{
    crossover::delta,
    genome::{Connection, Genome},
};
use rand::{rngs::ThreadRng, seq::IndexedRandom, Rng};
use std::{
    collections::HashMap,
    error::Error,
    hash::{DefaultHasher, Hash, Hasher},
};

pub struct InnoGen {
    pub head: usize,
    seen: HashMap<(usize, usize), usize>,
}

impl InnoGen {
    pub fn new(head: usize) -> Self {
        Self {
            head,
            seen: HashMap::new(),
        }
    }

    pub fn path(&mut self, v: (usize, usize)) -> usize {
        match self.seen.get(&v) {
            Some(n) => *n,
            None => {
                let n = self.head;
                self.head += 1;
                self.seen.insert(v, n);
                n
            }
        }
    }
}

#[derive(Debug)]
pub struct SpecieRepr<'a>(pub &'a [Connection]);

impl SpecieRepr<'_> {
    fn delta(&self, other: &[Connection]) -> f64 {
        delta(self.0, other)
    }
}

impl SpecieRepr<'_> {
    fn id(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

impl Hash for SpecieRepr<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for SpecieRepr<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for SpecieRepr<'_> {}

impl AsRef<[Connection]> for SpecieRepr<'_> {
    fn as_ref(&self) -> &[Connection] {
        self.0
    }
}

#[inline]
fn uniq_2<'a, T>(pool: &'a [T], rng: &mut ThreadRng) -> Option<(&'a T, &'a T)> {
    let len = pool.len();
    if len < 2 {
        None
    } else {
        let l = rng.random_range(0..len);
        let r = rng.random_range(0..len);
        if l == r {
            if r + 1 == len {
                Some((&pool[l], &pool[0]))
            } else {
                Some((&pool[l], &pool[r + 1]))
            }
        } else {
            Some((&pool[l], &pool[r]))
        }
    }
}

#[derive(Debug)]
pub struct Specie<'a>(pub SpecieRepr<'a>, pub Vec<(&'a Genome, usize)>);

impl Specie<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.1.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.1.is_empty()
    }

    #[inline]
    fn last(&self) -> Option<&(&Genome, usize)> {
        self.1.last()
    }

    pub fn fit_adjusted(&self) -> f64 {
        let l = self.len() as f64;
        self.1
            .iter()
            .fold(0., |acc, (_, fit)| acc + *fit as f64 / l)
    }

    fn reproduce_crossover(
        &self,
        size: usize,
        rng: &mut ThreadRng,
        innogen: &mut InnoGen,
    ) -> Result<Vec<Genome>, Box<dyn Error>> {
        if size == 0 {
            return Ok(vec![]);
        }

        if self.len() < 2 {
            return Err("too few members to crossover".into());
        }

        let mut pop = Vec::with_capacity(size);
        while pop.len() < size {
            let (l, r) = uniq_2(&self.1, rng).unwrap();
            let mut child = l.0.reproduce_with(r.0, l.1.cmp(&r.1), rng);
            child.maybe_mutate(rng, innogen)?;
            pop.push(child);
        }

        Ok(pop)
    }

    fn reproduce_copy(
        &self,
        size: usize,
        rng: &mut ThreadRng,
        innogen: &mut InnoGen,
    ) -> Result<Vec<Genome>, Box<dyn Error>> {
        if size == 0 {
            return Ok(vec![]);
        }

        if self.is_empty() {
            return Err("too few members to copy".into());
        }

        let mut pop = Vec::with_capacity(size);
        while pop.len() < size {
            let mut src = self.1.choose(rng).unwrap().0.clone();
            src.maybe_mutate(rng, innogen)?;
            pop.push(src);
        }

        Ok(pop)
    }

    pub fn reproduce(
        &self,
        size: usize,
        innogen: &mut InnoGen,
        rng: &mut ThreadRng,
    ) -> Result<Vec<Genome>, Box<dyn Error>> {
        if size == 0 {
            return Ok(vec![]);
        }

        if self.is_empty() {
            return Err("too few members to reproduce".into());
        }

        let mut pop: Vec<Genome> = Vec::with_capacity(size);
        pop.push(self.last().unwrap().0.clone());
        if size == 1 {
            return Ok(pop);
        }

        let size = size - 1;
        let size_copy = size / 4;
        let size_copy = if size_copy == 0 || self.len() == 1 {
            size
        } else {
            size_copy
        };

        // TODO reproduce_crossover and reproduce_copy can potentially be made faster
        // if they're handed a slice to write into intead of returning a vec that we then need to copy
        self.reproduce_copy(size_copy, rng, innogen)?
            .into_iter()
            .for_each(|genome| pop.push(genome));

        let size_crossover = size - size_copy;
        self.reproduce_crossover(size_crossover, rng, innogen)?
            .into_iter()
            .for_each(|genome| pop.push(genome));

        Ok(pop)
    }
}

const SPECIE_THRESHOLD: f64 = 4.;

pub fn speciate<'a>(genomes: impl Iterator<Item = (&'a Genome, usize)>) -> Vec<Specie<'a>> {
    let mut sp = Vec::new();
    for pair in genomes {
        match sp
            .iter_mut()
            .find(|Specie(repr, _)| repr.delta(&pair.0.connections) < SPECIE_THRESHOLD)
        {
            Some(Specie(_, members)) => members.push(pair),
            None => {
                sp.push(Specie(SpecieRepr(&pair.0.connections), vec![pair]));
            }
        }
    }

    for specie in sp.iter_mut() {
        specie.1.sort_by_cached_key(|(_, f)| *f);
    }

    sp
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

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

    #[test]
    fn test_uniq_2() {
        let mut rng = rng();
        assert_eq!(uniq_2::<usize>(&[], &mut rng), None);
        assert_eq!(uniq_2(&[&1], &mut rng), None);

        for _ in 0..10_000 {
            let (l, r) = uniq_2(&[1, 2], &mut rng).unwrap();
            if *l == 1 {
                assert_eq!(*r, 2);
            } else {
                assert_eq!(*r, 1);
                assert_eq!(*l, 2);
            }
        }

        let pool = (0..100).collect::<Vec<usize>>();
        for _ in 0..10_000 {
            let (l, r) = uniq_2(&pool, &mut rng).unwrap();
            assert_ne!(*l, *r)
        }
    }
}
