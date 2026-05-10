#![allow(confusable_idents)]
#![allow(mixed_script_confusables)]

use core::ops::ControlFlow;
use eevee::{
    genome::{Recurrent, WConnection},
    network::{activate::steep_sigmoid, Continuous, NonBias},
    playground::xor::{XorScenario, XOR_TARGET},
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    serialize::population_to_files,
    Connection, Genome, Stats,
};
use std::{fs, io::Write as _, path::PathBuf};

const POPULATION: usize = 100;
const GEN_LIMIT: usize = 500;

fn stop_hook<C: Connection, G: Genome<C>>(stats: &mut Stats<'_, C, G>) -> ControlFlow<()> {
    if stats.any_fitter_than(XOR_TARGET) || stats.generation >= GEN_LIMIT {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// Fixture generation macro
// ---------------------------------------------------------------------------
//
// For each (Connection, Genome, Network) triple, runs XOR evolution until the
// champion exceeds XOR_TARGET (or GEN_LIMIT) and saves the final population
// to  target/fixtures/xor/<C>_<G>_<N>/  as individual JSON files.
//
// Skips a permutation if the target directory already exists; delete it to
// force regeneration.

macro_rules! gen_fixture {
    (
        connections: [$($C:ident),+ $(,)?],
        genomes:     [$($G:ident),+ $(,)?],
        networks:    [$($N:ident),* $(,)?] $(,)?
    ) => {
        gen_fixture!(@foreach_c [$($C),+], [$($G),+], [$($N),*])
    };

    (@foreach_c [], $Gs:tt, $Ns:tt) => {};
    (@foreach_c [$C:ident $(, $Cs:ident)*], $Gs:tt, $Ns:tt) => {
        gen_fixture!(@foreach_g $C, $Gs, $Ns);
        gen_fixture!(@foreach_c [$($Cs),*], $Gs, $Ns);
    };

    (@foreach_g $C:ident, [], $Ns:tt) => {};
    (@foreach_g $C:ident, [$G:ident $(, $Gs:ident)*], $Ns:tt) => {
        gen_fixture!(@bench $C, $G, $Ns);
        gen_fixture!(@foreach_g $C, [$($Gs),*], $Ns);
    };

    (@bench $C:ident, $G:ident, [$($N:ident),*]) => {
        $(
            {
                let perm_id = concat!(stringify!($C), "_", stringify!($G), "_", stringify!($N));
                let dir = PathBuf::from("target/fixtures/xor").join(perm_id);

                if dir.exists() {
                    println!("{perm_id}: already exists, skipping (delete to regenerate)");
                } else {
                    print!("{perm_id}: evolving...");
                    std::io::stdout().flush().ok();

                    fs::create_dir_all(&dir)
                        .unwrap_or_else(|e| panic!("could not create {}: {e}", dir.display()));

                    let (species, _) = evolve(
                        XorScenario::<$N>::default(),
                        |(i, o)| population_init::<$C, $G<$C>>(i, o, POPULATION),
                        steep_sigmoid,
                        default_rng(),
                        EvolutionHooks::new(vec![Box::new(stop_hook::<$C, $G<$C>>)]),
                    );

                    population_to_files(&dir, &species)
                        .unwrap_or_else(|e| panic!("could not save {perm_id}: {e}"));

                    let count: usize = species.iter().map(|s| s.len()).sum();
                    let best = species
                        .iter()
                        .flat_map(|s| s.members.iter())
                        .map(|(_, f)| *f)
                        .fold(f64::NEG_INFINITY, f64::max);
                    println!(" {count} genomes, champion fitness {best:.4}");
                }
            }
        )*
    };
}

fn main() {
    gen_fixture!(
        connections: [WConnection],
        genomes:     [Recurrent],
        networks:    [Continuous, NonBias],
    );
}
