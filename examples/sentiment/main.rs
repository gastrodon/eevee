#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu,
    genome::{node::BTNode, Recurrent, WConnection},
    network::{Continuous, ToNetwork},
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    specie::{population_from_files, population_init, population_to_files},
    Connection, Genome, Network, Node, Scenario, Stats,
};
use core::f64;
use std::{fs::create_dir_all, ops::ControlFlow};

const POPULATION: usize = 1000;

fn decay_linear(want: f64, have: f64) -> f64 {
    if have.is_nan() {
        f64::MIN
    } else {
        0. - (want - have).abs()
    }
}

#[derive(Debug)]
enum SentimentKind {
    Positive,
    Negative,
}

impl SentimentKind {
    const fn value(&self) -> [f64; 2] {
        match self {
            SentimentKind::Positive => [1., 0.],
            SentimentKind::Negative => [0., 1.],
        }
    }
}

struct Sentiment<'a> {
    chunk_size: usize,
    data: Vec<(&'a str, Vec<Vec<f64>>, SentimentKind)>,
}

impl<'a> Sentiment<'a> {
    fn new(chunk_size: usize, positive: Vec<&'a str>, negative: Vec<&'a str>) -> Self {
        let plen = positive.len();
        let nlen = negative.len();
        let mut mixed = Vec::with_capacity(plen + nlen);
        for p in positive
            .iter()
            .map(|line| (*line, chunked(chunk_size, line), SentimentKind::Positive))
            .chain(
                negative
                    .iter()
                    .map(|line| (*line, chunked(chunk_size, line), SentimentKind::Negative)),
            )
        {
            mixed.push(p);
        }

        Self {
            chunk_size,
            data: mixed,
        }
    }
}

fn chunked(chunk_size: usize, data: &str) -> Vec<Vec<f64>> {
    let chunk_size = chunk_size * 8;
    let f_bits = data
        .as_bytes()
        .iter()
        .flat_map(|char| {
            (0..8)
                .map(|shift| match (char >> shift) & 1 {
                    0 => -1.,
                    1 => 1.,
                    b => unreachable!("invalid bit state {b} ({char} >> {shift} & 1)"),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    f_bits
        .chunks(chunk_size)
        .map(|chunk| {
            if chunk.len() < chunk_size {
                // maybe chunk.get(chunk_size-1) is faster? is len computed?
                let mut chunk_full = vec![0.; chunk_size];
                chunk_full[..chunk.len()].copy_from_slice(chunk);
                chunk_full
            } else {
                chunk.to_vec()
            }
        })
        .collect::<Vec<Vec<_>>>()
}

impl<
        'a,
        N: Node,
        C: Connection,
        G: Genome<N, C> + ToNetwork<Continuous, N, C>,
        A: Fn(f64) -> f64,
    > Scenario<N, C, G, A> for Sentiment<'a>
{
    fn io(&self) -> (usize, usize) {
        (8 * self.chunk_size, 2)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = genome.network();
        let fit = self
            .data
            .iter()
            .map(|(_, input, kind)| {
                for chunk in input {
                    network.step(5, chunk, σ);
                }

                let [w_positive, w_negative] = kind.value();
                let [c_positive, c_negative] = network.output() else {
                    unreachable!("incorrect output size")
                };

                decay_linear(w_positive, *c_positive) + decay_linear(w_negative, *c_negative)
            })
            .sum();

        fit
    }
}

fn hook<N: Node, C: Connection, G: Genome<N, C>>(
    stats: &mut Stats<'_, N, C, G>,
) -> ControlFlow<()> {
    let fittest = stats.fittest().unwrap();
    println!("fittest of gen {}: {:.4}", stats.generation, fittest.1);

    if stats.generation % 10 == 0 {
        population_to_files("output/sentiment", stats.species).unwrap();
    }

    if stats.generation == 250 {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    }
}

fn main() {
    let positive = include_str!("data/positive.txt").split('\n').collect();
    let negative = include_str!("data/negative.txt").split('\n').collect();

    type N = BTNode;
    type C = WConnection;
    type G = Recurrent<N, C>;

    create_dir_all("output/sentiment").expect("failed to create genome output");

    evolve(
        Sentiment::new(8, positive, negative),
        |(i, o)| {
            population_from_files("output/sentiment")
                .unwrap_or_else(|_| population_init::<N, C, G>(i, o, POPULATION))
        },
        relu,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
