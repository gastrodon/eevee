use rand::{rng, rngs::SmallRng, Rng, SeedableRng};
use std::{f64::consts::E, ops::BitXor};

pub fn steep_sigmoid(x: f64) -> f64 {
    1. / (1. + E.powf(-4.9 * x))
}

pub trait Game {
    /// generate and draw a new frame to the input slice of data
    fn frame(&self, data: &mut [f64]) -> bool;
    /// score the output slice of data against the last frame's expected result
    fn score(&self, data: &[f64]) -> Option<usize>;
    fn step(&mut self);
}

pub struct GameXOR(u8);

impl GameXOR {
    pub fn new() -> Self {
        Self(0)
    }
}

impl Game for GameXOR {
    fn frame(&self, data: &mut [f64]) -> bool {
        match self.0 {
            0 => {
                data[0] = 0.;
                data[1] = 0.;
                true
            }
            1 => {
                data[0] = 0.;
                data[1] = 1.;
                true
            }
            2 => {
                data[0] = 1.;
                data[1] = 0.;
                true
            }
            3 => {
                data[0] = 1.;
                data[1] = 1.;
                true
            }
            _ => false,
        }
    }

    fn score(&self, data: &[f64]) -> Option<usize> {
        let want = match self.0 {
            0 | 3 => 0.,
            1 | 2 => 1.,
            _ => return None,
        };

        Some((25. * (1. - (want - data[0]).abs())) as usize)
    }

    fn step(&mut self) {
        if self.0 < 4 {
            self.0 += 1
        }
    }
}
