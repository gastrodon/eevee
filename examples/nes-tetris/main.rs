#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use core::ops::ControlFlow;
use eevee::{
    activate::relu,
    genome::{Genome, Recurrent, WConnection},
    network::{Continuous, ToNetwork},
    population::population_init,
    random::default_rng,
    scenario::{evolve, EvolutionHooks},
    serialize_json::{population_from_files, population_to_files},
    Connection, Network, Scenario, Stats,
};
use nes_rust_slim::{
    button::Button, default_audio::DefaultAudio, default_display::DefaultDisplay,
    default_input::DefaultInput, rom::Rom, Nes,
};
use std::fs::create_dir_all;

#[allow(dead_code)]
#[rustfmt::skip]
mod v {
pub const ID: usize           = 0x42;
pub const ID_NEXT: usize      = 0x19;
pub const X: usize            = 0x40;
pub const Y: usize            = 0x41;
pub const SPEED: usize        = 0x44;
pub const FALLTIME: usize     = 0x45;
pub const GAME_MODE: usize    = 0xc0;
pub const GAME_OVER: usize    = 0x58;
pub const SEED_L: usize       = 0x17;
pub const SEED_R: usize       = 0x18;
pub const SCORE_1: usize      = 0x53;
pub const SCORE_2: usize      = 0x54;
pub const SCORE_3: usize      = 0x55;
pub const PIECE_COUNT: usize  = 0x1a;

pub const BOARD_OFFSET: usize = 0x400;
pub const BOARD_SIZE: usize   = 200;

pub const INPUT_SIZE: usize   = 200;

pub const PIECE_SHAPE: [[(u8, u8); 4]; 19] = [
    [(3, 2), (4, 1), (4, 2), (4, 3)], // T_UP
    [(1, 2), (2, 2), (2, 3), (3, 2)], // T_RIGHT
    [(2, 1), (2, 2), (2, 3), (3, 2)], // T_DOWN
    [(1, 2), (2, 1), (2, 2), (3, 2)], // T_LEFT
    [(1, 2), (2, 2), (3, 1), (3, 2)], // J_UP
    [(2, 1), (3, 1), (3, 2), (3, 3)], // J_RIGHT
    [(1, 2), (1, 3), (2, 2), (3, 2)], // J_DOWN
    [(2, 1), (2, 2), (2, 3), (3, 3)], // J_LEFT
    [(2, 1), (2, 2), (3, 2), (3, 3)], // Z_HORIZONTAL
    [(1, 3), (2, 2), (2, 3), (3, 2)], // Z_VERTICAL
    [(2, 1), (2, 2), (3, 1), (3, 2)], // O
    [(2, 2), (2, 3), (3, 1), (3, 2)], // S_HORIZONTAL
    [(1, 2), (2, 2), (2, 3), (3, 3)], // S_VERTICAL
    [(1, 2), (2, 2), (3, 2), (3, 3)], // L_UP
    [(2, 1), (2, 2), (2, 3), (3, 1)], // L_RIGHT
    [(1, 1), (1, 2), (2, 2), (3, 2)], // L_DOWN
    [(2, 3), (3, 1), (3, 2), (3, 3)], // L_LEFT
    [(0, 2), (1, 2), (2, 2), (3, 2)], // I_VERTICAL
    [(3, 0), (3, 1), (3, 2), (3, 3)], // I_HORIZONTAL
];
}

use v::*;
fn sense_board(ram: &[u8], sense: &mut [f64; INPUT_SIZE]) {
    *sense = [0.; INPUT_SIZE];
    for (idx, _) in ram[BOARD_OFFSET..BOARD_OFFSET + BOARD_SIZE]
        .iter()
        .enumerate()
        .filter(|(_, b)| **b != 0xef)
    {
        sense[idx] = 1.;
    }

    if (0..19).contains(&ram[ID]) {
        for index in PIECE_SHAPE[ram[ID] as usize]
            .iter()
            .filter_map(|(row, col)| {
                let row = row + ram[Y];
                let col = col + ram[X];
                (row >= 2 && col >= 2).then(|| ((row - 2) as usize * 10) + (col - 2) as usize)
            })
            .filter(|index| *index < 200)
        {
            sense[index] = -1.;
        }
    } else {
        // TODO what piece am I missing??????
    }
}

fn score(ram: &[u8]) -> f64 {
    // real score | piece count
    (((ram[SCORE_1] as usize) << 8)
        | ((ram[SCORE_2] as usize) << 16)
        | ((ram[SCORE_3] as usize) << 24)
        | (ram[PIECE_COUNT] as usize)) as f64
}

#[cfg(feature = "watch_game")]
mod watch {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    pub static GENERATION: AtomicUsize = AtomicUsize::new(0);
    pub static SPECIES: AtomicUsize = AtomicUsize::new(0);
    pub static MAX_FITNESS: AtomicU64 = AtomicU64::new(0);

    pub fn update(generation: usize, species: usize, max_fitness: f64) {
        GENERATION.store(generation, Ordering::Relaxed);
        SPECIES.store(species, Ordering::Relaxed);
        MAX_FITNESS.store(max_fitness.to_bits(), Ordering::Relaxed);
    }

    pub fn read() -> (usize, usize, f64) {
        (
            GENERATION.load(Ordering::Relaxed),
            SPECIES.load(Ordering::Relaxed),
            f64::from_bits(MAX_FITNESS.load(Ordering::Relaxed)),
        )
    }
}

#[cfg(feature = "watch_game")]
fn draw_footer(current_score: f64) {
    let (gen, _, _) = watch::read();
    let left = format!("{}", current_score as u64);
    let right = format!("{}", gen);
    let inner = (left.len() + 1 + right.len()).max(8);
    let spaces = inner - left.len() - right.len();
    println!("|{}|", "-".repeat(inner));
    println!("|{}{}{}|", left, " ".repeat(spaces), right);
}

#[cfg(feature = "watch_game")]
fn draw_buttons(buttons: &[bool; 8]) {
    // (display_char, button_index) — indices from NES joypad order: 0=A 1=B 2=Sel 3=Start 4=Up 5=Down 6=Left 7=Right
    const MAP: [(char, usize); 8] = [
        ('a', 0),
        ('b', 1),
        ('^', 4),
        ('<', 6),
        ('>', 7),
        ('.', 5),
        ('!', 3),
        ('#', 2),
    ];
    let mut row = String::new();
    for (i, (ch, idx)) in MAP.iter().enumerate() {
        if i > 0 {
            row.push(' ');
        }
        row.push(if buttons[*idx] { *ch } else { ' ' });
    }
    println!("{}", row);
}

#[cfg(feature = "watch_game")]
fn draw_sense(sense: &[f64; INPUT_SIZE]) {
    for chunk in sense.chunks(10) {
        println!(
            "{}",
            chunk
                .iter()
                .map(|data| match data {
                    -1. => '-',
                    0. => '_',
                    1. => '+',
                    _ => '?',
                })
                .collect::<String>()
        )
    }
}

fn enter_game(nes: &mut Nes) {
    while nes.get_cpu().get_ram().data[0xc3] == 0 {
        nes.step_frame();
    }
    nes.get_mut_cpu().get_mut_ram().data[0xc3] = 0;
    while nes.get_cpu().get_ram().data[GAME_MODE] == 0 {
        nes.step_frame();
    }
    while nes.get_cpu().get_ram().data[GAME_MODE] != 4 {
        nes.press_button(Button::Start);
        nes.step_frame();
        nes.release_button(Button::Start);
        nes.step_frame();
    }

    nes.get_mut_cpu().get_mut_ram().data[SEED_L] = 0;
    nes.get_mut_cpu().get_mut_ram().data[SEED_R] = 0;
}

#[cfg(feature = "watch_game")]
fn run_exhibition_game(genome: &Recurrent<WConnection>) {
    let mut nes = Nes::new(
        Box::new(DefaultInput::new()),
        Box::new(DefaultDisplay::new()),
        Box::new(DefaultAudio::new()),
    );
    nes.set_rom(Rom::new(include_bytes!("data/tetris.nes").to_vec()));
    nes.bootup();
    enter_game(&mut nes);

    let mut network: Continuous = genome.network();
    let mut sense = [0.; INPUT_SIZE];
    while nes.get_cpu().get_ram().data[GAME_OVER] == 0 {
        sense_board(&nes.get_cpu().get_ram().data, &mut sense);
        network.step(1, &sense, relu);

        for (idx, x) in network.output().iter().enumerate() {
            if idx == 2 || idx == 3 {
                continue;
            }
            nes.get_mut_cpu().joypad1.buttons[idx] = *x >= 0.5;
        }
        let buttons = nes.get_cpu().joypad1.buttons;
        nes.step_frame();

        print!("\x1b[H");
        draw_sense(&sense);
        let current_score = score(&nes.get_cpu().get_ram().data);
        draw_footer(current_score);
        draw_buttons(&buttons);

        nes.get_mut_cpu().joypad1.buttons = [false; 8];
    }
}

struct NesTetris;

impl<C: Connection, G: Genome<C> + ToNetwork<Continuous, C>, A: Fn(f64) -> f64> Scenario<C, G, A>
    for NesTetris
{
    fn io(&self) -> (usize, usize) {
        (200, 8)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut nes = Nes::new(
            Box::new(DefaultInput::new()),
            Box::new(DefaultDisplay::new()),
            Box::new(DefaultAudio::new()),
        );
        nes.set_rom(Rom::new(include_bytes!("data/tetris.nes").to_vec()));
        nes.bootup();
        enter_game(&mut nes);

        let mut network = genome.network();
        let mut sense = [0.; 200];
        while nes.get_cpu().get_ram().data[GAME_OVER] == 0 {
            sense_board(&nes.get_cpu().get_ram().data, &mut sense);
            network.step(1, &sense, σ);

            for (idx, x) in network.output().iter().enumerate() {
                if idx == 2 || idx == 3 {
                    continue;
                }
                nes.get_mut_cpu().joypad1.buttons[idx] = *x >= 0.5;
            }
            nes.step_frame();
            nes.get_mut_cpu().joypad1.buttons = [false; 8];
        }

        score(&nes.get_cpu().get_ram().data)
    }
}

const POPULATION: usize = 100;

fn main() {
    type C = WConnection;
    type G = Recurrent<C>;

    create_dir_all("output/nes-tetris").expect("failed to create genome output");

    let init = population_from_files("output/nes-tetris")
        .unwrap_or_else(|_| population_init::<C, G>(INPUT_SIZE, 8, POPULATION));

    #[cfg(feature = "watch_game")]
    let best: std::sync::Arc<std::sync::Mutex<Option<G>>> = {
        let seed = init
            .0
            .first()
            .and_then(|s| s.members.first())
            .map(|(g, _)| g.clone());
        std::sync::Arc::new(std::sync::Mutex::new(seed))
    };

    #[cfg(feature = "watch_game")]
    {
        let slot = std::sync::Arc::clone(&best);
        std::thread::spawn(move || {
            print!("\x1b[2J\x1b[H");
            loop {
                let genome = slot.lock().unwrap().clone();
                match genome {
                    Some(g) => run_exhibition_game(&g),
                    None => std::thread::sleep(std::time::Duration::from_millis(50)),
                }
            }
        });
    }

    #[cfg(feature = "watch_game")]
    let hook_best = std::sync::Arc::clone(&best);

    let hook = move |stats: &mut Stats<'_, C, G>| -> ControlFlow<()> {
        #[cfg(feature = "watch_game")]
        {
            let max = stats.fittest().map(|(_, f)| *f).unwrap_or(0.0);
            watch::update(stats.generation, stats.species.len(), max);
            if let Some((genome, _)) = stats.species.first().and_then(|s| s.members.first()) {
                *hook_best.lock().unwrap() = Some(genome.clone());
            }
        }

        if !stats.generation.is_multiple_of(10) {
            return ControlFlow::Continue(());
        }

        let fittest = stats.fittest().unwrap();
        println!("gen {} best: {:.3}", stats.generation, fittest.1);
        population_to_files("output/nes-tetris", stats.species).unwrap();

        if stats.generation == 400 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    };

    evolve(
        NesTetris {},
        |_| init,
        relu,
        default_rng(),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
