#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu,
    genome::CTRGenome,
    random::{default_rng, ProbBinding, ProbStatic},
    scenario::{evolve, EvolutionHooks},
    specie::population_init,
    Genome, Happens, Network, Probabilities, Scenario, Stats,
};
use core::f64;
use rand::RngCore;
use std::ops::ControlFlow;

const POPULATION: usize = 100;

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

struct Sentiment {
    chunk_size: usize,
    positive: Vec<&'static str>,
    negative: Vec<&'static str>,
}

fn decay_linear(want: f64, have: f64) -> f64 {
    0. - (want - have).abs()
}

fn chunked(chunk_size: usize, data: &'static str) -> Vec<Vec<f64>> {
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

impl<G: Genome, H: RngCore + Probabilities + Happens, A: Fn(f64) -> f64> Scenario<G, H, A>
    for Sentiment
{
    fn io(&self) -> (usize, usize) {
        (8 * self.chunk_size, 2)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let inputs = {
            let plen = self.positive.len();
            let nlen = self.negative.len();
            let mut mixed = Vec::with_capacity(plen + nlen);
            for p in self
                .positive
                .iter()
                .map(|s| (s, chunked(self.chunk_size, s), SentimentKind::Positive))
                .chain(
                    self.negative
                        .iter()
                        .map(|s| (s, chunked(self.chunk_size, s), SentimentKind::Negative)),
                )
            {
                mixed.push(p);
            }

            mixed
        };

        let mut network = genome.network();
        let fit = inputs
            .into_iter()
            .map(|(_, input, kind)| {
                for chunk in input {
                    network.step(5, &chunk, σ);
                }

                let [w_positive, w_negative] = kind.value();
                let [c_positive, c_negative] = {
                    let [c_positive, c_negative] = network.output() else {
                        unreachable!("incorrect output size")
                    };
                    [c_positive.clamp(-1., 1.), c_negative.clamp(0., 100.)]
                };

                decay_linear(w_positive, c_positive) + decay_linear(w_negative, c_negative)
            })
            .sum();

        fit
    }
}

fn hook<
    G: Genome,
    H: RngCore + Probabilities<Update = (brain::random::EvolutionEvent, u64)> + Happens,
>(
    stats: &mut Stats<'_, G, H>,
) -> ControlFlow<()> {
    let fittest = stats.fittest().unwrap();
    println!("fittest of gen {}: {:.4}", stats.generation, fittest.1);

    if stats.generation == 100 {
        fittest
            .0
            .to_file(format!("output/sentiment-{}.json", stats.generation))
            .unwrap();
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    }
}

fn main() {
    let positive = include_str!("positive.txt").split('\n').collect();
    let negative = include_str!("negative.txt").split('\n').collect();

    evolve(
        Sentiment {
            chunk_size: 8,
            positive,
            negative,
        },
        |(i, o)| population_init::<CTRGenome>(i, o, POPULATION),
        relu,
        ProbBinding::new(ProbStatic::default(), default_rng()),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
