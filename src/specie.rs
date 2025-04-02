use crate::{
    crossover::delta,
    genome::{Connection, Genome},
    random::Happens,
    Node,
};
use core::{error::Error, f64};
use fxhash::FxHashMap;
use rand::RngCore;
use std::{
    collections::HashMap,
    fs::read_dir,
    hash::{DefaultHasher, Hash, Hasher},
    iter::empty,
    marker::PhantomData,
    path::Path,
};

pub struct InnoGen {
    pub head: usize,
    seen: FxHashMap<(usize, usize), usize>,
}

impl InnoGen {
    pub fn new(head: usize) -> Self {
        Self {
            head,
            seen: FxHashMap::default(),
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

#[derive(Debug, Clone)]
pub struct SpecieRepr<N: Node, C: Connection<N>>(Vec<C>, PhantomData<N>);

impl<N: Node, C: Connection<N>> SpecieRepr<N, C> {
    fn new(v: Vec<C>) -> Self {
        Self(v, PhantomData)
    }

    fn delta(&self, other: &[C]) -> f64 {
        delta(&self.0, other)
    }

    #[inline]
    fn cloned(&self) -> Vec<C> {
        self.0.to_vec()
    }
}

impl<N: Node, C: Connection<N>> SpecieRepr<N, C> {
    fn id(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

impl<N: Node, C: Connection<N>> Hash for SpecieRepr<N, C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<N: Node, C: Connection<N>> PartialEq for SpecieRepr<N, C> {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<N: Node, C: Connection<N>> Eq for SpecieRepr<N, C> {}

impl<N: Node, C: Connection<N>> AsRef<[C]> for SpecieRepr<N, C> {
    fn as_ref(&self) -> &[C] {
        &self.0
    }
}

#[derive(Debug)]
pub struct Specie<N: Node, C: Connection<N>, G: Genome<N, C>> {
    pub repr: SpecieRepr<N, C>,
    pub members: Vec<(G, f64)>,
}

impl<N: Node, C: Connection<N>, G: Genome<N, C>> Specie<N, C, G> {
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

fn reproduce_crossover<N: Node, C: Connection<N>, G: Genome<N, C>, H: RngCore + Happens>(
    genomes: &[(G, f64)],
    size: usize,
    rng: &mut H,
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

fn reproduce_copy<N: Node, C: Connection<N>, G: Genome<N, C>, H: RngCore + Happens>(
    genomes: &[(G, f64)],
    size: usize,
    rng: &mut H,
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

pub fn reproduce<N: Node, C: Connection<N>, G: Genome<N, C>, H: RngCore + Happens>(
    genomes: Vec<(G, f64)>,
    size: usize,
    innogen: &mut InnoGen,
    rng: &mut H,
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
    pop.push(
        genomes
            .iter()
            .max_by(|(_, l), (_, r)| {
                l.partial_cmp(r)
                    .unwrap_or_else(|| panic!("cannot partial_cmp {l} and {r}"))
            })
            .unwrap()
            .0
            .clone(),
    );

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

/// allocate a target population for every specie in an existing population
fn population_alloc<'a, N: Node + 'a, C: Connection<N> + 'a, G: Genome<N, C> + 'a>(
    species: impl Iterator<Item = &'a Specie<N, C, G>>,
    population: usize,
) -> HashMap<SpecieRepr<N, C>, usize> {
    let species_fitted = species
        .map(|s| (s.repr.clone(), s.fit_adjusted()))
        .collect::<Vec<_>>();

    let fit_total = species_fitted.iter().fold(0., |acc, (_, n)| acc + n);
    let population_f = population as f64;
    species_fitted
        .into_iter()
        .map(|(specie_repr, fit_adjusted)| {
            (
                specie_repr,
                f64::round(population_f * fit_adjusted / fit_total) as usize,
            )
        })
        .collect()
}

pub type SpecieGroup<N, C, G> = (Vec<Specie<N, C, G>>, usize);

/// initial population of a single specie consisting of single connection genomes
/// while it's not necessarily recommended to do an initual mutation, it allows us to mutate a
/// bisection on any genome without the need to check for existing connections beforehand
pub fn population_init<N: Node, C: Connection<N>, G: Genome<N, C>>(
    sensory: usize,
    action: usize,
    population: usize,
) -> SpecieGroup<N, C, G> {
    let (genome, inno_head) = G::new(sensory, action);
    (
        vec![Specie {
            repr: SpecieRepr::new(genome.connections().to_vec()),
            members: vec![(genome, f64::MIN); population],
        }],
        inno_head,
    )
}

pub fn population_to_files<P: AsRef<Path>, N: Node, C: Connection<N>, G: Genome<N, C>>(
    path: P,
    pop: &[Specie<N, C, G>],
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

pub fn population_from_files<P: AsRef<Path>, N: Node, C: Connection<N>, G: Genome<N, C>>(
    path: P,
) -> Result<SpecieGroup<N, C, G>, Box<dyn Error>> {
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

fn population_allocated<
    'a,
    N: Node + 'a,
    C: Connection<N> + 'a,
    G: Genome<N, C> + 'a,
    T: Iterator<Item = &'a (Specie<N, C, G>, f64)>,
>(
    species: T,
    population: usize,
) -> impl Iterator<Item = (Vec<(G, f64)>, usize)> {
    let viable = species
        .filter_map(|(specie, min_fitness)| {
            let viable = specie
                .members
                .iter()
                .filter(|&pair| (&pair.1 >= min_fitness))
                .cloned()
                .collect::<Vec<_>>();

            // (!viable.is_empty()).then_some((&specie.repr, viable));
            (!viable.is_empty()).then(|| Specie {
                repr: specie.repr.clone(),
                members: viable,
            })
        })
        .collect::<Vec<_>>();

    let alloc = population_alloc(viable.iter(), population);

    viable
        .into_iter()
        .filter_map(move |specie| alloc.get(&specie.repr).map(|pop| (specie.members, *pop)))
}

// reproduce a whole speciated population into a non-speciated population
pub fn population_reproduce<N: Node, C: Connection<N>, G: Genome<N, C>, H: RngCore + Happens>(
    species: &[(Specie<N, C, G>, f64)],
    population: usize,
    inno_head: usize,
    rng: &mut H,
) -> (Vec<G>, usize) {
    // let species = population_viable(species.into_iter());
    // let species_pop = population_alloc(species, population);
    let mut innogen = InnoGen::new(inno_head);
    (
        population_allocated(species.iter(), population)
            .flat_map(|(members, pop)| reproduce(members, pop, &mut innogen, rng).unwrap())
            .collect::<Vec<_>>(),
        innogen.head,
    )
}

const SPECIE_THRESHOLD: f64 = 4.;

pub fn speciate<N: Node, C: Connection<N>, G: Genome<N, C>>(
    genomes: impl Iterator<Item = (G, f64)>,
    reprs: impl Iterator<Item = SpecieRepr<N, C>>,
) -> Vec<Specie<N, C, G>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        genome::{node::NonBNode, CTRGenome, WConnection},
        random::{default_rng, ProbBinding, ProbStatic},
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

    type BasicGenomeCtrnn = CTRGenome<NonBNode, WConnection<NonBNode>>;

    test_t!(specie_init[T: BasicGenomeCtrnn]() {
        let count = 40;
        let (species, inno_head) = population_init::<NonBNode, WConnection<NonBNode>, T>(2, 2, count);
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

    test_t!(specie_reproduce[T: BasicGenomeCtrnn]() {
        let mut rng = ProbBinding::new(ProbStatic::default(), default_rng());
        let count = 40;
        let (species, inno_head) = population_init::<NonBNode, WConnection<NonBNode>, T>(2, 2, count);

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
}
