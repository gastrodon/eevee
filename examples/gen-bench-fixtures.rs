//! Regenerate bench fixture files under benches/data/.
//! Run with: cargo run --example gen-bench-fixtures --features serialize_json

use eevee::{
    genome::{Recurrent, WConnection},
    population::population_init,
    serialize::SerializeFile,
};

type C = WConnection;
type G = Recurrent<C>;

fn main() {
    // ctr-genome-xor-100.json: 100 (genome, fitness) pairs, 2 sensory 1 action
    let (species, _) = population_init::<C, G>(2, 1, 100);
    let xor: Vec<(G, f64)> = species
        .into_iter()
        .flat_map(|s| s.members)
        .collect();
    let json = serde_json::to_string_pretty(&xor).unwrap();
    std::fs::write("benches/data/ctr-genome-xor-100.json", json).unwrap();
    println!("wrote ctr-genome-xor-100.json ({} genomes)", xor.len());

    // ctr-genome-rand-100.json: single genome, 6 sensory 6 action
    let (species, _) = population_init::<C, G>(6, 6, 1);
    let genome = species.into_iter().flat_map(|s| s.members).next().unwrap().0;
    std::fs::write(
        "benches/data/ctr-genome-rand-100.json",
        genome.to_str().unwrap(),
    )
    .unwrap();
    println!("wrote ctr-genome-rand-100.json");
}
