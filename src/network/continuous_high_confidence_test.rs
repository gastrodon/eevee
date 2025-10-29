//! Test demonstrating that CTRNNs can produce high confidence outputs (>0.9)
//! This addresses the concern about whether valid genome configurations exist
//! that can produce networks capable of high confidence signals.

#[cfg(test)]
mod high_confidence_tests {
    use crate::{
        activate::steep_sigmoid,
        genome::{InnoGen, Genome, Recurrent, WConnection},
        network::{Continuous, FromGenome, Network},
        Connection,
    };

    #[test]
    fn test_high_confidence_via_strong_bias() {
        // Create a genome with a strong bias-to-output connection
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = Recurrent::<WConnection>::new(1, 1);
        
        // Add strong bias connection
        let mut conn = WConnection::new(2, 1, &mut inno);
        conn.weight = 10.0;
        genome.push_connection(conn);
        
        let mut network = Continuous::from_genome(&genome);
        
        // Test with various inputs
        for input_val in [0.0, 0.5, 1.0] {
            network.flush();
            for _ in 0..50 {
                network.step(10, &[input_val], steep_sigmoid);
            }
            
            let output = network.output()[0];
            assert!(
                output > 0.9,
                "Expected output > 0.9, got {} for input {}",
                output,
                input_val
            );
        }
    }

    #[test]
    fn test_high_confidence_via_self_loop() {
        // Create a genome with input connection and self-reinforcing loop
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = Recurrent::<WConnection>::new(1, 1);
        
        let mut conn1 = WConnection::new(0, 1, &mut inno);
        conn1.weight = 5.0;
        genome.push_connection(conn1);
        
        let mut conn2 = WConnection::new(1, 1, &mut inno);
        conn2.weight = 2.5;
        genome.push_connection(conn2);
        
        let mut network = Continuous::from_genome(&genome);
        
        // Test with high input values
        for input_val in [0.7, 0.8, 0.9, 1.0] {
            network.flush();
            for _ in 0..50 {
                network.step(10, &[input_val], steep_sigmoid);
            }
            
            let output = network.output()[0];
            assert!(
                output > 0.9,
                "Expected output > 0.9, got {} for input {}",
                output,
                input_val
            );
        }
    }

    #[test]
    fn test_high_confidence_controlled_range() {
        // Create a genome with moderate weights for more controlled output
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = Recurrent::<WConnection>::new(1, 1);
        
        let mut conn1 = WConnection::new(0, 1, &mut inno);
        conn1.weight = 2.0;
        genome.push_connection(conn1);
        
        let mut conn2 = WConnection::new(2, 1, &mut inno);
        conn2.weight = 1.5;
        genome.push_connection(conn2);
        
        let mut network = Continuous::from_genome(&genome);
        
        // With fewer steps, output should still exceed 0.9
        network.flush();
        for _ in 0..5 {
            network.step(5, &[1.0], steep_sigmoid);
        }
        
        let output = network.output()[0];
        assert!(
            output > 0.9,
            "Expected output > 0.9, got {}",
            output
        );
    }

    #[test]
    fn test_multiple_outputs_high_confidence() {
        // Create a genome with 2 inputs and 2 outputs
        let mut inno = InnoGen::new(0);
        let (mut genome, _) = Recurrent::<WConnection>::new(2, 2);
        
        // Strong connections from bias to both outputs
        let mut conn1 = WConnection::new(4, 2, &mut inno); // bias to output1
        conn1.weight = 8.0;
        genome.push_connection(conn1);
        
        let mut conn2 = WConnection::new(4, 3, &mut inno); // bias to output2
        conn2.weight = 8.0;
        genome.push_connection(conn2);
        
        let mut network = Continuous::from_genome(&genome);
        
        network.flush();
        for _ in 0..50 {
            network.step(10, &[0.5, 0.5], steep_sigmoid);
        }
        
        let outputs = network.output();
        assert!(
            outputs[0] > 0.9 && outputs[1] > 0.9,
            "Expected both outputs > 0.9, got [{}, {}]",
            outputs[0],
            outputs[1]
        );
    }
}
