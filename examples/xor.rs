#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use core::{f64, ops::ControlFlow};
use eevee::{
    genome::{Genome, Recurrent, WConnection},
    network::{activate::steep_sigmoid, Continuous, Network, ToNetwork},
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    serialize::SerializeFile,
    Connection, Scenario, Stats,
};

const POPULATION: usize = 1000;

struct Xor;

macro_rules! eval_pair {
    ($pair:expr, $want:expr, ($network:ident $fit:ident $σ:ident)) => {{
        $network.step(20, &$pair, $σ);
        // tanh maps unbounded y → (-1, 1); targets are ±1
        let v = $network.output()[0].tanh();
        let error = ($want - v).abs(); // max error is 2.0 (e.g. want=1, v≈-1)
        $fit += 1.0 - 0.5 * error;    // scale so worst case = 0.0, perfect = 1.0
        $network.flush();
    }};
}

impl<C: Connection, G: Genome<C> + ToNetwork<Continuous, C>, A: Fn(f64) -> f64> Scenario<C, G, A>
    for Xor
{
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = genome.network();
        let mut fit = 0.;

        eval_pair!([0., 0.],  1., (network fit σ));
        eval_pair!([1., 1.],  1., (network fit σ));
        eval_pair!([1., 0.], -1., (network fit σ));
        eval_pair!([0., 1.], -1., (network fit σ));

        fit
    }
}

fn dump_generation<C: Connection, G: Genome<C> + SerializeFile>(stats: &Stats<'_, C, G>) {
    use std::io::Write;

    let gen = stats.generation;
    let dir = format!("output/{gen}");
    std::fs::create_dir_all(&dir).unwrap();

    let all: Vec<(&G, f64)> = stats
        .species
        .iter()
        .flat_map(|s| s.members.iter().map(|(g, f)| (g, *f)))
        .collect();

    // Champion
    if let Some((champ, _)) = stats.fittest() {
        champ.to_file(format!("{dir}/genome-champ.json")).unwrap();
    }

    // 10 evenly-spaced samples across the population
    let n = all.len();
    for i in 0..10usize {
        let idx = (i * n) / 10;
        if let Some((genome, _)) = all.get(idx) {
            genome.to_file(format!("{dir}/genome-{i}.json")).unwrap();
        }
    }

    // CSV row
    let csv_path = "output/run.csv";
    let write_header = !std::path::Path::new(csv_path).exists();
    let mut csv = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)
        .unwrap();

    if write_header {
        writeln!(csv, "generation,best_fitness,best_nodes,best_conns,species,population,mean_fitness").unwrap();
    }

    let (best_nodes, best_conns, best_fitness) = stats
        .fittest()
        .map(|(g, f)| (g.nodes().len(), g.connections().len(), *f))
        .unwrap_or((0, 0, 0.));
    let mean_fitness = if all.is_empty() {
        0.
    } else {
        all.iter().map(|(_, f)| f).sum::<f64>() / all.len() as f64
    };

    writeln!(
        csv,
        "{gen},{best_fitness:.6},{best_nodes},{best_conns},{},{},{mean_fitness:.6}",
        stats.species.len(),
        all.len(),
    )
    .unwrap();
}

fn hook<C: Connection, G: Genome<C> + SerializeFile>(stats: &mut Stats<'_, C, G>) -> ControlFlow<()> {
    if stats.generation % 10 == 0 {
        dump_generation(stats);
    }

    if stats.generation % 10 == 0 {
        let (g, f) = stats.fittest().unwrap();
        println!(
            "gen {}: {:.4} ({} nodes, {} conns) of {} species",
            stats.generation,
            f,
            g.nodes().len(),
            g.connections().len(),
            stats.species.len()
        );
    }

    if stats.any_fitter_than(3.9) {
        println!("target met in gen {}", stats.generation);
        return ControlFlow::Break(());
    }

    if stats.generation >= 500 {
        println!("generation limit reached");
        return ControlFlow::Break(());
    }

    ControlFlow::Continue(())
}

type C = WConnection;
type G = Recurrent<C>;

fn main() {
    evolve(
        Xor {},
        |(i, o)| population_init::<C, G>(i, o, POPULATION),
        steep_sigmoid,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
