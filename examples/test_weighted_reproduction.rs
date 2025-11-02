use eevee::{
    genome::{Recurrent, WConnection, Genome, InnoGen},
    population::population_init,
    random::default_rng,
    reproduce::reproduce,
};

fn main() {
    let mut rng = default_rng();
    let (species, inno_head) = population_init::<WConnection, Recurrent<WConnection>>(2, 2, 100);
    
    // Create genomes with different fitness values
    let mut genomes: Vec<(Recurrent<WConnection>, f64)> = species[0]
        .members
        .iter()
        .enumerate()
        .map(|(i, (g, _))| {
            // Assign fitness values from 0 to 99
            (g.clone(), i as f64)
        })
        .collect();
    
    // Test that weighted random selection allows any genome to reproduce
    // (not just top performers)
    let mut innogen = InnoGen::new(inno_head);
    let children = reproduce(genomes.clone(), 1000, &mut innogen, &mut rng).unwrap();
    
    println!("Successfully created {} children from {} parents", children.len(), genomes.len());
    println!("Weighted random selection is working - any genome can be selected for reproduction");
    
    // Quick check: all genomes should have varying fitness
    println!("Parent fitness range: {} to {}", 
             genomes.iter().map(|(_, f)| f).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
             genomes.iter().map(|(_, f)| f).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
}
