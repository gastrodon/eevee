#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use approx::relative_eq;
use core::{f64, ops::ControlFlow};
use eevee::{
    activate::relu,
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
        // Log species info
        let num_species = stats.species.len();
        let species_info: Vec<String> = stats
            .species
            .iter()
            .enumerate()
            .filter(|(_, specie)| !specie.members.is_empty())
            .map(|(idx, specie)| {
                let size = specie.len();
                let avg_fitness = specie.members.iter().map(|(_, f)| f).sum::<f64>() / size as f64;
                let max_fitness = specie.members.iter().map(|(_, f)| f).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let min_fitness = specie.members.iter().map(|(_, f)| f).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let adjusted_fitness = specie.fit_adjusted();
                format!(
                    "S{}: size={}, avg_fit={:.2}, max_fit={:.2}, min_fit={:.2}, adj_fit={:.2}",
                    idx, size, avg_fitness, max_fitness, min_fitness, adjusted_fitness
                )
            })
            .collect();

        let log_line = format!(
            "Gen {}: {} species | {}\n",
            stats.generation,
            num_species,
            species_info.join(" | ")
        );
        
        let _ = log_file.borrow_mut().write_all(log_line.as_bytes());
        
        if stats.generation % 10 == 0 || stats.generation < 20 {
            print!("{}", log_line);
        }

        if stats.generation >= 100 {
            println!("Stopping after 100 generations for analysis");
            return ControlFlow::Break(());
        }

        if stats.any_fitter_than(400. - f64::EPSILON) {
            let fittest = stats.fittest().unwrap();
            println!("target met in gen {}: {:.4}", stats.generation, fittest.1);
            let _ = fittest
                .0
                .to_file(format!("output/xor-diagnostic-{}.json", stats.generation));
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }
}

type C = WConnection;
type G = Recurrent<C>;

fn main() {
    let log_file = RefCell::new(File::create("output/species_diagnostic.log").expect("Failed to create log file"));
    
    evolve(
        Xor {},
        |(i, o)| population_init::<C, G>(i, o, POPULATION),
        relu,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook(log_file))]),
    );
}
