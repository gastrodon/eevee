use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

#[derive(Debug)]
pub struct Network<T: Fn(f64) -> f64 + Sized> {
    pub σ: T,           // activation function                  (\u3c3)
    pub y: Matrix<f64>, // 1d state of neurons 0-N
    pub θ: Matrix<f64>, // 1d bias of neurons 0-N               (\u3b8)
    pub τ: Matrix<f64>, // 1d membrane resistance time constant (\u3c4)
    pub w: Matrix<f64>, // Nd weights between neurons, indexed as [from, to]
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl<T: Fn(f64) -> f64> Network<T> {
    pub fn new(size: usize, σ: T, sensory: (usize, usize), action: (usize, usize)) -> Self {
        Self {
            σ,
            y: Matrix::zeros(1, size),
            θ: Matrix::zeros(1, size),
            τ: Matrix::ones(1, size),
            w: Matrix::zeros(size, size),
            sensory,
            action,
        }
    }

    pub fn step(&mut self, prec: usize, input: &[f64]) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            self.y += (((&self.y + &self.θ).apply(&self.σ) * &self.w) - &self.y + &m_input)
                .elediv(&self.τ)
                .apply(&|v| v * inv);
        }
    }

    pub fn state(&self) -> &[f64] {
        self.y.data()
    }
}
