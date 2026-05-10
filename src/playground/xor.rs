use crate::{network::ToNetwork, Connection, Genome, Network, Scenario};
use core::marker::PhantomData;
use rand::Rng;

/// All permutations of xor for 2 bits
pub const XOR_PAIRS: [([f64; 2], f64); 4] = [
    ([0., 0.], 1.),
    ([1., 1.], 1.),
    ([1., 0.], -1.),
    ([0., 1.], -1.),
];

/// Fitness at which XOR is considered solved.
pub const XOR_TARGET: f64 = 9.99;

/// Scenario that evaluates a genome on 10 randomly sampled XOR pairs.
/// Parameterised over the network type so it can be reused across different
/// network architectures without duplicating eval logic.
pub struct XorScenario<NN>(PhantomData<NN>);

impl<NN> Default for XorScenario<NN> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C, G, A, NN> Scenario<C, G, A> for XorScenario<NN>
where
    C: Connection,
    G: Genome<C> + ToNetwork<NN, C>,
    A: Fn(f64) -> f64,
    NN: Network,
{
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = <G as ToNetwork<NN, C>>::network(genome);
        let mut fit = 0.0;
        for _ in 0..10 {
            let (input, want) = XOR_PAIRS[rand::rng().random_range(0..4)];
            network.step(&input, σ);
            let v = network.output()[0].tanh();
            fit += 1.0 - 0.5 * (want - v).abs();
        }
        fit
    }
}
