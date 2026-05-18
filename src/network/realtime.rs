use super::{default_prec, Continuous, FromGenome};
use crate::{Connection, Genome, Network, Recurrent};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

/// A stateful recurrent NN for realtime / continuous-time problems.
///
/// Based on the CTRNN dynamics described in:
/// "On the dynamics of small continuous-time recurrent neural networks" (Beer 1995).
#[derive(Debug, Clone)]
pub struct Realtime {
    /// Integration steps per [Network::step] call.
    pub prec: usize,
    /// 1d state of neurons 0-N
    pub y: Matrix<f64>,
    /// 1d bias of neurons 0-N
    pub θ: Matrix<f64>,
    /// 1d membrane resistance time constant
    pub τ: Matrix<f64>,
    /// Nd weights between neurons, indexed as [from, to]
    pub w: Matrix<f64>,
    /// Range of input neurons, indexing into y
    pub sensory: (usize, usize),
    /// Range of output neurons, indexing into y
    pub action: (usize, usize),
}

impl Network for Realtime {
    fn step<F: Fn(f64) -> f64>(&mut self, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (self.prec as f64);
        for _ in 0..self.prec {
            self.y += (((&self.y + &self.θ).apply(&σ) * &self.w) - &self.y + &m_input)
                .elemul(&self.τ)
                .apply(&|v| v * inv);
        }
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}

impl crate::Recurrent for Realtime {}

impl Continuous for Realtime {
    fn reset(&mut self) {
        self.y = Matrix::zeros(1, self.y.cols());
    }
}

impl<C: Connection, G: Genome<C> + Recurrent> FromGenome<C, G> for Realtime {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.node_count();
        Self {
            prec: default_prec(),
            y: Matrix::zeros(1, cols),
            θ: Matrix::zeros(1, cols),
            τ: Matrix::new(1, cols, vec![1.0; cols]),
            w: {
                let mut w = vec![0.; cols * cols];
                for c in genome.connections().iter().filter(|c| c.enabled()) {
                    w[c.from() * cols + c.to()] = c.weight();
                }
                Matrix::new(cols, cols, w)
            },
            sensory: (genome.sensory().start, genome.sensory().end),
            action: (genome.action().start, genome.action().end),
        }
    }
}

#[derive(Debug)]
pub struct RealtimeUnbias {
    /// Integration steps per [Network::step] call.
    pub prec: usize,
    pub y: Matrix<f64>,
    pub w: Matrix<f64>,
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl Network for RealtimeUnbias {
    fn step<F: Fn(f64) -> f64>(&mut self, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (self.prec as f64);
        for _ in 0..self.prec {
            self.y = ((&self.y + &m_input).apply(&σ) * &self.w).apply(&|v| v * inv);
        }
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}

impl Recurrent for RealtimeUnbias {}

impl Continuous for RealtimeUnbias {
    fn reset(&mut self) {
        self.y = Matrix::zeros(1, self.y.cols());
    }
}

impl<C: Connection, G: Genome<C> + Recurrent> FromGenome<C, G> for RealtimeUnbias {
    fn from_genome(genome: &G) -> Self {
        let cols = genome.node_count();
        Self {
            prec: default_prec(),
            y: Matrix::zeros(1, cols),
            w: {
                let mut w = vec![0.; cols * cols];
                for c in genome.connections().iter().filter(|c| c.enabled()) {
                    w[c.from() * cols + c.to()] = c.weight();
                }
                Matrix::new(cols, cols, w)
            },
            sensory: (genome.sensory().start, genome.sensory().end),
            action: (genome.action().start, genome.action().end),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        assert_f64_approx,
        genome::{connection::BWConnection, Genome, InnoGen, Recurrent, WConnection},
    };
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,
        G: Recurrent<C>,

        #[test]
        fn test_from_genome() {
            let mut inno = InnoGen::new(0);
            let (mut genome, _) = G::new(2, 2);
            genome.push_connection(C::new(0, 3, &mut inno));
            genome.push_connection(C::new(0, 1, &mut inno));
            genome.push_connection(C::new(0, 1, &mut inno));

            let nn = Realtime::from_genome(&genome);
            unsafe {
                for c in genome.connections() {
                    if c.enabled() {
                        assert_f64_approx!(nn.w.get_unchecked([c.from(), c.to()]), c.weight());
                    }
                }

                for i in 0..genome.node_count() {
                    assert_f64_approx!(nn.θ.get_unchecked([0, i]), 0.)
                }
            }

            assert_eq!(
                (nn.sensory.0, nn.sensory.1),
                (genome.sensory().start, genome.sensory().end)
            );
            assert_eq!(
                (nn.action.0, nn.action.1),
                (genome.action().start, genome.action().end)
            );
        }

        /// behavior consistency: realtime network with hidden neurons responds to input
        #[test]
        fn test_realtime_behavior_consistent() {
            // Direct construction with hidden neurons (6, 7).
            // Nodes: 0-2 sensory, 3-5 action, 6-7 hidden
            let n = 8;
            let mut w = vec![0.; n * n];

            // Set weights as [from, to] in row-major order
            w[0 * n + 6] = 0.5;  // sensory[0] → hidden[0]
            w[6 * n + 3] = 0.5;  // hidden[0] → action[0]
            w[6 * n + 4] = 0.5;  // hidden[0] → action[1]
            w[1 * n + 3] = 0.5;  // sensory[1] → action[0]
            w[1 * n + 7] = 0.5;  // sensory[1] → hidden[1]
            w[7 * n + 5] = 0.5;  // hidden[1] → action[2]
            w[2 * n + 4] = 0.5;  // sensory[2] → action[1]
            w[2 * n + 5] = 0.5;  // sensory[2] → action[2]

            let mut nn = Realtime {
                prec: 10,
                y: Matrix::zeros(1, n),
                θ: Matrix::zeros(1, n),
                τ: Matrix::new(1, n, vec![1.0; n]),
                w: Matrix::new(n, n, w),
                sensory: (0, 3),
                action: (3, 6),
            };

            // Test statefulness: same input produces different outputs across steps
            let cases: [(&[f64], [[f64; 3]; 3]); 3] = [
                (&[1.0, 0.5, -0.5], [
                    [0.4543099041996693, 0.31011019901830134, 0.3033087902498634],
                    [0.7206136447449458, 0.4270440385315579, 0.4145301830023249],
                    [0.8393129308972523, 0.4721907947994719, 0.46050742149690216],
                ]),
                (&[-1.0, 0.5, 1.0], [
                    [0.4174063200035319, 0.4582208483881388, 0.48832302381583825],
                    [0.6067921532596537, 0.655314229873935, 0.756621865829994],
                    [0.6763042841417588, 0.7202325609154627, 0.8715578343683869],
                ]),
                (&[0.0, 1.0, -1.0], [
                    [0.47775761240772074, 0.25192885045713903, 0.2692956706336944],
                    [0.7214650462220639, 0.33085128678011355, 0.3785219619172767],
                    [0.8203001141077999, 0.36532142446261123, 0.42826251802576787],
                ]),
            ];

            for (case_i, (input, expected_states)) in cases.iter().enumerate() {
                nn.reset();
                for (step_i, expected) in expected_states.iter().enumerate() {
                    nn.step(input, crate::activate::steep_sigmoid);
                    let output = nn.output();
                    assert_eq!(output.len(), 3);
                    assert_eq!(output, expected.as_slice(), "case {}, step {}: input: {:?}, got: {:?}", case_i, step_i, input, output);
                }
            }
        }
    }
}
