use crate::network::Ctrnn;
use rand_distr::num_traits::Pow;
use std::f64::consts::E;

pub fn steep_sigmoid(x: f64) -> f64 {
    1. / (1. + E.powf(-4.9 * x))
}

pub trait Game<T: Fn(f64) -> f64 + Sized> {
    fn eval(&self, network: &mut Ctrnn<T>) -> usize;
}

pub struct GameXOR;

impl GameXOR {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T: Fn(f64) -> f64 + Sized> Game<T> for GameXOR {
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
