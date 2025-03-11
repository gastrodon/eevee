#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu,
    network::loss::{decay_linear, decay_quadratic},
    specie::population_init,
    Ctrnn, EvolutionTarget, Genome, Network, Scenario, Specie,
};
use core::f64;

const POPULATION: usize = 100;

#[derive(Debug)]
enum SentimentKind {
    Positive,
    Negative,
}

impl SentimentKind {
    const fn value(&self) -> [f64; 2] {
        match self {
            SentimentKind::Positive => [100., 0.],
            SentimentKind::Negative => [0., 100.],
        }
    }
}

struct Sentiment {
    chunk_size: usize,
    positive: Vec<&'static str>,
    negative: Vec<&'static str>,
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

impl Scenario for Sentiment {
    fn io(&self) -> (usize, usize) {
        (8 * self.chunk_size, 2)
    }

    fn eval<F: Fn(f64) -> f64>(&self, genome: &Genome, σ: F) -> f64 {
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
                .take(1)
            {
                mixed.push(p);
            }

            mixed
        };

        let mut network = Ctrnn::from_genome(genome);
        let fit = inputs
            .into_iter()
            .take(10)
            .map(|(input_raw, input, kind)| {
                for chunk in input {
                    network.step(5, &chunk, &σ);
                }

                let [w_positive, w_negative] = kind.value();
                let [c_positive, c_negative] = {
                    let [c_positive, c_negative] = network.output() else {
                        unreachable!("incorrect output size")
                    };
                    [c_positive.clamp(-1., 1.), c_negative.clamp(0., 100.)]
                };

                // println!(
                //     "{input_raw}\n\tpos:{c_positive} ({})\n\tneg:{c_negative} ({})",
                //     decay_linear(w_positive, c_positive),
                //     decay_linear(w_negative, c_negative)
                // );
                decay_linear(w_positive, c_positive) + decay_linear(w_negative, c_negative)
            })
            .sum();

        fit
    }
}

fn main() {
    let scenario = Sentiment {
        chunk_size: 8,
        positive: include_str!("positive.txt").split('\n').collect(),
        negative: include_str!("negative.txt").split('\n').collect(),
    };

    let res = scenario.evolve(
        EvolutionTarget::Generation(4000),
        |(i, o)| population_init(i, o, POPULATION),
        POPULATION,
        relu,
    );

    println!(
        "top score: {:?}",
        res.0
            .into_iter()
            .flat_map(|Specie { members, .. }| members.into_iter().map(|(_, fit)| fit))
            .collect::<Vec<_>>(),
        // .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
        // .unwrap()
        // .1
    );
}
