use std::{f64::consts::E, ops::BitXor};

use rand::{rng, rngs::SmallRng, Rng, SeedableRng};

pub fn steep_sigmoid(x: f64) -> f64 {
    1. / (1. + E.powf(-4.9 * x))
}

pub trait Game {
    /// generate and draw a new frame to the input slice of data
    fn frame(&mut self, data: &mut [f64]) -> bool;
    /// score the output slice of data against the last frame's expected result
    fn score(&self, data: &[f64]) -> Option<usize>;
}

pub struct GameXOR {
    iters: usize,
    input_current: Option<(f64, f64)>,
    rng: SmallRng,
}

impl GameXOR {
    pub fn new(iters: usize) -> Self {
        Self {
            iters,
            input_current: None,
            rng: SmallRng::from_rng(&mut rng()),
        }
    }
}

fn xor_f64(l: f64, r: f64) -> f64 {
    assert!(l == 1. || l == 0.);
    assert!(r == 1. || r == 0.);
    if (l == 1.).bitxor(r == 1.) {
        1.
    } else {
        0.
    }
}

// distance between w and h that exponentially decays, scaled to 100
fn xor_score(w: f64, h: f64) -> usize {
    (100. * (1. + -(w - h).abs().powf(0.4))) as usize
}

impl Game for GameXOR {
    fn frame(&mut self, data: &mut [f64]) -> bool {
        assert!(data.len() == 2);
        if self.iters != 0 {
            self.iters -= 1;
            let (l, r) = (
                if self.rng.random_bool(0.5) { 1. } else { 0. },
                if self.rng.random_bool(0.5) { 1. } else { 0. },
            );
            data[0] = l;
            data[1] = r;
            self.input_current = Some((l, r));
            true
        } else {
            false
        }
    }

    /// I wonder if we should score linearly for success / fail
    /// or score based on how _close_ the genome is to the correct xor
    /// probably the latter, since the first would stifle inperfect innovation
    /// but also should score exponentialy higher as we approach whole correct f64s
    fn score(&self, data: &[f64]) -> Option<usize> {
        self.input_current
            .map(|(l, r)| xor_score(xor_f64(l, r), steep_sigmoid(data[0])))
    }
}
