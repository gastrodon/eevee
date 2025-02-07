mod crossover;
mod eval;
mod genome;

use crate::genome::Genome;

fn main() {
    let genome_xor = Genome::new(2, 1);
    println!("{genome_xor:?}")
}
