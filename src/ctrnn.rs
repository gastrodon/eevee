use rulinalg::matrix::Matrix;

const STEP_RATE: usize = 1;
const STEP_SIZE: usize = 1;

/// for every synaptic pair j -> i, describes a anetwork
struct Network {
    state: Matrix<f64>,          // y - 1d state of neurons 0-N
    input: Matrix<f64>,          // I - 1d external input to neurons 0-N
    bias: Matrix<f64>,           // θ - 1d bias of neurons 0-N
    time_const: Matrix<f64>,     // τ - 1d membrane resistance time constant ( τ > 0 )
    weight_from_to: Matrix<f64>, // w - Nd weights between neurons (0-N, 0-N), 0. if disconnected
}
