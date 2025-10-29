#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use approx::relative_eq;
use core::{f64, ops::ControlFlow};
use eevee::{
    activate::relu,
    crossover::delta,
    genome::{Genome, Recurrent, WConnection},
    network::{Network, Simple, ToNetwork},
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    Connection, Scenario, Stats,
};
use std::cell::RefCell;
use std::fs::File;
use std::io::Write;

const POPULATION: usize = 1000;

struct Xor;

macro_rules! eval_pair {
    ($pair:expr, $want:expr, ($network:ident $fit:ident $σ:ident)) => {{
        $network.step(2, &$pair, $σ);
        let v = $network.output()[0];
        if relative_eq!(v, $want, epsilon = 0.05) {
            $fit += 100.;
        } else if (-1. ..=2.).contains(&v) {
            $fit -= ($want - v).abs();
        } else {
            $fit -= v.abs() * v.abs();
        }
        $network.flush();
    }};
}

impl<C: Connection, G: Genome<C> + ToNetwork<Simple<C>, C>, A: Fn(f64) -> f64> Scenario<C, G, A>
    for Xor
{
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = genome.network();
        let mut fit = 0.;

        eval_pair!([0., 0.], 1., (network fit σ));
        eval_pair!([1., 1.], 1., (network fit σ));
        eval_pair!([1., 0.], 0., (network fit σ));
        eval_pair!([0., 1.], 0., (network fit σ));

        fit
    }
}

fn hook<C: Connection, G: Genome<C>>(
    log_file: RefCell<File>,
) -> impl Fn(&mut Stats<'_, C, G>) -> ControlFlow<()> {
    move |stats: &mut Stats<'_, C, G>| -> ControlFlow<()> {
        // Sample some genomes and compute deltas
        let all_genomes: Vec<&G> = stats
            .species
            .iter()
            .flat_map(|s| s.members.iter().map(|(g, _)| g))
            .collect();
        
        if all_genomes.len() >= 10 && stats.generation % 10 == 0 {
            // Sample 10 pairs
            let mut deltas = Vec::new();
            for i in 0..5 {
                let g1 = all_genomes[i * 2];
                let g2 = all_genomes[i * 2 + 1];
                let d = delta(g1.connections(), g2.connections());
                deltas.push(d);
            }
            let avg_delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
            let max_delta = deltas.iter().cloned().fold(0./0., f64::max);
            
            let log_line = format!(
                "Gen {}: {} species, sample deltas: avg={:.4}, max={:.4}, values={:?}\n",
                stats.generation,
                stats.species.len(),
                avg_delta,
                max_delta,
                deltas.iter().map(|d| format!("{:.2}", d)).collect::<Vec<_>>()
            );
            
            let _ = log_file.borrow_mut().write_all(log_line.as_bytes());
            print!("{}", log_line);
        }

        if stats.generation >= 50 {
            println!("Stopping after 50 generations");
            return ControlFlow::Break(());
        }

        if stats.any_fitter_than(400. - f64::EPSILON) {
            let fittest = stats.fittest().unwrap();
            println!("Target met in gen {}: {:.4}", stats.generation, fittest.1);
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }
}

type C = WConnection;
type G = Recurrent<C>;

fn main() {
    let log_file = RefCell::new(File::create("output/delta_analysis.log").expect("Failed to create log file"));
    
    evolve(
        Xor {},
        |(i, o)| population_init::<C, G>(i, o, POPULATION),
        relu,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook(log_file))]),
    );
}
