use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

pub mod activate {
    use std::f64::consts::E;

    pub fn steep_sigmoid(x: f64) -> f64 {
        1. / (1. + E.powf(-4.9 * x))
    }

    pub fn relu(x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            x
        }
    }
}

pub trait Network {
    fn step(&mut self, prec: usize, input: &[f64]);
    fn output(&self) -> &[f64];
}

#[derive(Debug)]
pub struct Ctrnn<T: Fn(f64) -> f64 + Sized> {
    pub σ: T,           // activation function                  (\u3c3)
    pub y: Matrix<f64>, // 1d state of neurons 0-N
    pub θ: Matrix<f64>, // 1d bias of neurons 0-N               (\u3b8)
    pub τ: Matrix<f64>, // 1d membrane resistance time constant (\u3c4)
    pub w: Matrix<f64>, // Nd weights between neurons, indexed as [from, to]
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl<T: Fn(f64) -> f64 + Sized> Network for Ctrnn<T> {
    fn step(&mut self, prec: usize, input: &[f64]) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            self.y += (((&self.y + &self.θ).apply(&self.σ) * &self.w) - &self.y + &m_input)
                .elediv(&self.τ)
                .apply(&|v| v * inv);
        }
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}
