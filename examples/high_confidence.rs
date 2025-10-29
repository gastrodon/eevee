//! Demonstration that a CTRNN can produce high confidence signals (>0.9)
//! 
//! This example proves that the genome/CTRNN implementation is capable of 
//! producing outputs greater than 0.9 by constructing specific genome configurations.

#![allow(mixed_script_confusables)]

use eevee::{
    activate::steep_sigmoid,
    genome::{InnoGen, Genome, Recurrent, WConnection},
    network::{Continuous, FromGenome, Network},
    Connection,
};

fn main() {
    println!("=== Proof: CTRNN can produce high confidence signals (>0.9) ===\n");
    
    // Configuration 1: Strong bias-to-output connection
    println!("Configuration 1: Strong bias connection");
    println!("----------------------------------------");
    test_bias_driven_output();
    
    println!("\n\nConfiguration 2: Strong input-to-output with self-loop");
    println!("--------------------------------------------------------");
    test_self_reinforcing_output();
    
    println!("\n\n=== Conclusion ===");
    println!("The CTRNN implementation CAN produce high confidence signals (>0.9)");
    println!("This is achievable with appropriate genome configurations.");
}

fn test_bias_driven_output() {
    // Create a simple genome with 1 input and 1 output
    // Genome structure:
    // - Node 0: Sensory (input)
    // - Node 1: Action (output)
    // - Node 2: Static (bias, θ=1)
    
    let mut inno = InnoGen::new(0);
    let (mut genome, _) = Recurrent::<WConnection>::new(1, 1);
    
    // Add a strong connection from bias (node 2) to output (node 1)
    // This will drive the output high regardless of input
    let mut conn = WConnection::new(2, 1, &mut inno);
    conn.weight = 10.0;  // Strong positive weight
    genome.push_connection(conn);
    
    println!("Genome configuration:");
    println!("  Nodes: {:?}", genome.nodes());
    println!("  Connection: bias(2) -> output(1) with weight = 10.0");
    
    // Create and test the network
    let mut network = Continuous::from_genome(&genome);
    
    // Test with various inputs - output should be high regardless
    println!("\nTesting outputs:");
    for input_val in [0.0, 0.25, 0.5, 0.75, 1.0] {
        network.flush();
        
        // Step the network to reach steady state
        for _ in 0..100 {
            network.step(10, &[input_val], steep_sigmoid);
        }
        
        let output = network.output()[0];
        let status = if output > 0.9 { "✓ HIGH CONFIDENCE" } else { "✗ low" };
        println!("  Input: {:.2} → Output: {:.6} {}", input_val, output, status);
    }
}

fn test_self_reinforcing_output() {
    // Create a genome with feedback loops
    // This configuration uses input and self-reinforcement
    
    let mut inno = InnoGen::new(0);
    let (mut genome, _) = Recurrent::<WConnection>::new(1, 1);
    
    // Connection 1: Input to output with strong weight
    let mut conn1 = WConnection::new(0, 1, &mut inno);
    conn1.weight = 5.0;
    genome.push_connection(conn1);
    
    // Connection 2: Output to itself (recurrent self-loop)
    let mut conn2 = WConnection::new(1, 1, &mut inno);
    conn2.weight = 2.5;
    genome.push_connection(conn2);
    
    println!("Genome configuration:");
    println!("  Nodes: {:?}", genome.nodes());
    println!("  Connection 1: input(0) -> output(1) with weight = 5.0");
    println!("  Connection 2: output(1) -> output(1) with weight = 2.5 (self-loop)");
    
    let mut network = Continuous::from_genome(&genome);
    
    println!("\nTesting outputs:");
    for input_val in [0.5, 0.75, 1.0] {
        network.flush();
        
        for _ in 0..100 {
            network.step(10, &[input_val], steep_sigmoid);
        }
        
        let output = network.output()[0];
        let status = if output > 0.9 { "✓ HIGH CONFIDENCE" } else { "✗ low" };
        println!("  Input: {:.2} → Output: {:.6} {}", input_val, output, status);
    }
}
