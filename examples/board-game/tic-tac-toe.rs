#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use core::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use board_game::board::{Board, Outcome, Player};
use board_game::games::ttt::TTTBoard;
use board_game::util::coord::Coord3;
use rand::{Rng, RngCore};

use eevee::{
    genome::{Genome, Recurrent, WConnection},
    network::{activate::steep_sigmoid, Continuous, Network, ToNetwork},
    population::population_init,
    random::{seed_urandom, WyRng},
    scenario::{evolve, EvolutionHooks},
    Scenario, Stats,
};

const POPULATION: usize = 500;
const GAMES_PER_EVAL: usize = 8;
const HALL_OF_FAME_MAX: usize = 64;
const HALL_REFRESH_EVERY: usize = 1;
const NETWORK_PREC: usize = 20;
const MAX_GENERATIONS: usize = 200;

type C = WConnection;
type G = Recurrent<C>;

fn coord(i: usize) -> Coord3 {
    Coord3::from_xy((i % 3) as u8, (i / 3) as u8)
}

fn encode_board(board: &TTTBoard, viewpoint: Player) -> [f64; 18] {
    let mut out = [0.0f64; 18];
    for i in 0..9 {
        match board.tile(coord(i)) {
            Some(p) if p == viewpoint => out[i] = 1.0,
            Some(_) => out[i + 9] = 1.0,
            None => {}
        }
    }
    out
}

fn legal_moves(board: &TTTBoard) -> Vec<Coord3> {
    if board.is_done() {
        return vec![];
    }
    (0..9)
        .map(coord)
        .filter(|c| board.is_available_move(*c).unwrap_or(false))
        .collect()
}

fn network_move<A: Fn(f64) -> f64>(
    network: &mut Continuous,
    board: &TTTBoard,
    viewpoint: Player,
    σ: &A,
) -> Option<Coord3> {
    let legal = legal_moves(board);
    if legal.is_empty() {
        return None;
    }
    let input = encode_board(board, viewpoint);
    network.step(NETWORK_PREC, &input, σ);
    let output = network.output();

    let mut best = legal[0];
    let mut best_score = output[(best.x() + 3 * best.y()) as usize];
    for &c in &legal[1..] {
        let score = output[(c.x() + 3 * c.y()) as usize];
        if score > best_score {
            best = c;
            best_score = score;
        }
    }
    Some(best)
}

fn random_move<R: RngCore>(board: &TTTBoard, rng: &mut R) -> Option<Coord3> {
    let legal = legal_moves(board);
    if legal.is_empty() {
        None
    } else {
        Some(legal[rng.random_range(0..legal.len())])
    }
}

fn play_game<A: Fn(f64) -> f64>(
    learner: &mut Continuous,
    learner_player: Player,
    opponent: Option<&mut Continuous>,
    σ: &A,
    rng: &mut WyRng,
) -> f64 {
    let mut board = TTTBoard::default();
    learner.flush();
    let mut opponent = opponent;
    if let Some(o) = opponent.as_deref_mut() {
        o.flush();
    }

    while !board.is_done() {
        let mover = board.next_player();
        let mv = if mover == learner_player {
            network_move(learner, &board, learner_player, σ)
        } else {
            match opponent.as_deref_mut() {
                Some(o) => network_move(o, &board, mover, σ),
                None => random_move(&board, rng),
            }
        };
        match mv {
            Some(m) => {
                board.play(m).expect("legal move filter should have prevented this");
            }
            None => break,
        }
    }

    match board.outcome() {
        Some(Outcome::WonBy(p)) if p == learner_player => 1.0,
        Some(Outcome::Draw) => 0.5,
        _ => 0.0,
    }
}

struct TttScenario {
    pool: Arc<RwLock<Vec<G>>>,
    seed_counter: AtomicU64,
}

impl TttScenario {
    fn new(pool: Arc<RwLock<Vec<G>>>, base_seed: u64) -> Self {
        Self {
            pool,
            seed_counter: AtomicU64::new(base_seed),
        }
    }
}

impl<A: Fn(f64) -> f64> Scenario<C, G, A> for TttScenario {
    fn io(&self) -> (usize, usize) {
        (18, 9)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let seed = self.seed_counter.fetch_add(1, Ordering::Relaxed);
        let mut rng = WyRng::seeded(seed);

        let opponents: Vec<G> = {
            let pool = self.pool.read().unwrap();
            if pool.is_empty() {
                vec![]
            } else {
                (0..GAMES_PER_EVAL)
                    .map(|_| pool[rng.random_range(0..pool.len())].clone())
                    .collect()
            }
        };

        let mut learner = genome.network();
        let mut total = 0.0;

        for i in 0..GAMES_PER_EVAL {
            let learner_player = if i % 2 == 0 { Player::A } else { Player::B };
            let score = if opponents.is_empty() {
                play_game(&mut learner, learner_player, None, σ, &mut rng)
            } else {
                let mut opp_net = opponents[i % opponents.len()].network();
                play_game(&mut learner, learner_player, Some(&mut opp_net), σ, &mut rng)
            };
            total += score;
        }

        total / GAMES_PER_EVAL as f64
    }
}

fn refresh_hook(pool: Arc<RwLock<Vec<G>>>) -> Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>> {
    Box::new(move |stats| {
        if stats.generation % HALL_REFRESH_EVERY == 0 {
            let overall_champ: Option<G> =
                stats.fittest().map(|(g, _)| g.clone());

            if let Some(champ) = overall_champ {
                let mut pool = pool.write().unwrap();
                pool.push(champ);
                let drop_n = pool.len().saturating_sub(HALL_OF_FAME_MAX);
                if drop_n > 0 {
                    pool.drain(0..drop_n);
                }
            }
        }
        ControlFlow::Continue(())
    })
}

fn report_hook(pool: Arc<RwLock<Vec<G>>>) -> Box<dyn Fn(&mut Stats<'_, C, G>) -> ControlFlow<()>> {
    Box::new(move |stats| {
        if stats.generation % 5 == 0 {
            if let Some((g, f)) = stats.fittest() {
                let hall_size = pool.read().unwrap().len();
                let mean_fitness = mean_fitness(stats);
                println!(
                    "gen {}: best {:.4} mean {:.4} ({} nodes, {} conns) | {} species | hall {}",
                    stats.generation,
                    f,
                    mean_fitness,
                    g.node_count(),
                    g.connections().len(),
                    stats.species.len(),
                    hall_size
                );
            }
        }
        if stats.generation >= MAX_GENERATIONS {
            println!("generation limit {} reached", MAX_GENERATIONS);
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    })
}

fn mean_fitness<C: eevee::Connection, G: Genome<C>>(stats: &Stats<'_, C, G>) -> f64 {
    let (sum, n) = stats
        .species
        .iter()
        .flat_map(|s| s.members.iter())
        .fold((0.0, 0usize), |(s, n), (_, f)| (s + *f, n + 1));
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn main() {
    let pool: Arc<RwLock<Vec<G>>> = Arc::new(RwLock::new(Vec::new()));
    let base_seed = seed_urandom().unwrap();
    let scenario = TttScenario::new(Arc::clone(&pool), base_seed);

    evolve(
        scenario,
        |(i, o)| population_init::<C, G>(i, o, POPULATION),
        steep_sigmoid,
        WyRng::seeded(base_seed.wrapping_add(0xdead_beef)),
        EvolutionHooks::new(vec![
            refresh_hook(Arc::clone(&pool)),
            report_hook(Arc::clone(&pool)),
        ]),
    );
}
