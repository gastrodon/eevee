use crate::network::Ctrnn;
use rand_distr::num_traits::Pow;

pub trait Scenario<T: Fn(f64) -> f64 + Sized> {
    fn eval(&self, network: &mut Ctrnn<T>) -> usize;
}

pub struct ConstXOR;

impl ConstXOR {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T: Fn(f64) -> f64 + Sized> Scenario<T> for ConstXOR {
    fn eval(&self, network: &mut Ctrnn<T>) -> usize {
        let mut fit = 0;
        network.step(2, &[0., 0.]);
        fit += (25. * (1. - (1. - network.output()[0]).abs().pow(2.))) as usize;

        network.step(2, &[1., 1.]);
        fit += (25. * (1. - (1. - network.output()[0]).abs().pow(2.))) as usize;

        network.step(2, &[0., 1.]);
        fit += (25. * (1. - (0. - network.output()[0]).abs().pow(2.))) as usize;

        network.step(2, &[1., 0.]);
        fit += (25. * (1. - (0. - network.output()[0]).abs().pow(2.))) as usize;

        fit
    }
}
