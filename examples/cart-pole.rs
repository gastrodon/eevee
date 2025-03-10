use brain::{activate::steep_sigmoid, specie::population_init, EvolutionTarget, Genome, Scenario};
use gym_rs::{
    core::Env, envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode,
};
use rand::rng;

pub struct CartPole {
    evals: usize,
    render: RenderMode,
}

impl CartPole {
    fn new(evals: usize, render: RenderMode) -> Self {
        Self { evals, render }
    }
}

impl Scenario for CartPole {
    fn io() -> (usize, usize) {
        (4, 2)
    }

    fn eval<T: Fn(f64) -> f64>(&self, genome: &Genome, σ: T) -> f64 {
        let mut env = CartPoleEnv::new(self.render);
        let mut fit = 0.;
        let mut act = 0;
        let mut network = Ctrnn::from_genome(genome);
        for _ in 0..self.evals {
            network.step(3, &Into::<Vec<f64>>::into(env.state), &σ);

            let out = network.output();
            if out[0] >= 0.8 && out[0] > out[1] {
                act = 0;
            } else if out[1] >= 0.8 {
                act = 1;
            }

            let res = env.step(act);
            fit += res.reward.into_inner();
            if res.done {
                break;
            }
        }
        fit
    }
}
const POPULATION: usize = 100;

/// requires sdl2_gfx
fn main() {
    let champ = CartPole::new(2500, RenderMode::None)
        .evolve(
            EvolutionTarget::Fitness(100.),
            |(i, o)| population_init(i, o, POPULATION),
            POPULATION,
            steep_sigmoid,
        )
        .0
        .into_iter()
        .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
        .unwrap();

    println!("champ fit: {}", champ.1);
    CartPole::new(2500, RenderMode::Human).eval(&mut champ.0.network(), steep_sigmoid);
}
