//! Functions and structs related to RNG, random events, and mutation probabilities.

use core::cmp::min;
use find_fold::FindFold;
use rand::RngCore;
use std::{
    fs::File,
    io::{self, Read},
    ops::ControlFlow,
};

use crate::events;

/// A function for turning a whole percent into a u64 that is `x` percent of `[u64::MAX]`. This
/// is useful because when we calculate RNGs, we get a raw u64. If our rolled u64 is less than
/// the probability u64, the RNG check passes
///
/// # Examples
///
/// ```
/// use eevee::random::{percent, default_rng};
/// use rand::RngCore;
///
/// const ONE_PERCENT: u64 = u64::MAX / 100;
///
/// assert_eq!(percent(50), 50 * ONE_PERCENT);
/// assert_eq!(percent(1), ONE_PERCENT);
///
/// // This will pass about 10% of the time!
/// default_rng().next_u64() < percent(10);
/// ```
pub const fn percent(x: u64) -> u64 {
    x * (u64::MAX / 100)
}

/// A quick and dirty way to get an RNG seed from urandom, onsystems that support it. Useful
/// because our implementation of WyRng always needs a seed
pub fn seed_urandom() -> io::Result<u64> {
    let mut file = File::open("/dev/urandom")?;
    let mut buffer = [0u8; 8];
    file.read_exact(&mut buffer)?;
    Ok(u64::from_le_bytes([
        buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
    ]))
}

/// Fast seed function using current timestamp
pub fn seed_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    nanos as u64
}

/// Fast seed function using process ID and timestamp
pub fn seed_pid_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = std::process::id() as u128;
    let combined = nanos.wrapping_mul(pid).wrapping_add(nanos);
    combined as u64
}

/// Fast seed function using thread ID and timestamp
pub fn seed_thread_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::thread;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let thread_id = thread::current().id();
    let thread_hash = format!("{:?}", thread_id).bytes().fold(0u64, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(b as u64)
    });
    let combined = nanos.wrapping_mul(thread_hash as u128).wrapping_add(nanos);
    combined as u64
}

/// For getting a handle on an RngCore when you don't want to think too much about it. This is
/// why Eevee doesn't work on Windows.
pub fn default_rng() -> impl RngCore {
    WyRng::seeded(seed_urandom().unwrap())
}

/// A really small but also fast random number generator. Lifted from smol-rs/fastrand
pub struct WyRng {
    state: u64,
}

impl WyRng {
    pub fn seeded(state: u64) -> Self {
        Self { state }
    }
}

impl RngCore for WyRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        const WY_CONST_0: u64 = 0x2d35_8dcc_aa6c_78a5;
        const WY_CONST_1: u64 = 0x8bb8_4b93_962e_acc9;
        self.state = self.state.wrapping_add(WY_CONST_0);
        let t = u128::from(self.state) * u128::from(self.state ^ WY_CONST_1);
        (t as u64) ^ (t >> 64) as u64
    }

    // TODO test this
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }

        let mut idx = 0;
        while idx < dst.len() {
            let lim = min(8, dst.len() - idx);
            dst.copy_from_slice(&self.next_u64().to_ne_bytes()[..lim]);
            idx += lim;
        }
    }
}

/// A struct for describing discrete events that may occur, typically related to what mutation
/// happens when any mutation is invoked. Mostly here so that we can use
pub trait EventKind: Copy {
    const COUNT: usize;
    fn variants() -> [Self; Self::COUNT];
    fn idx(&self) -> usize;

    fn pick<R: RngCore>(rng: &mut R, prob: [u64; Self::COUNT]) -> Option<Self> {
        let roll = rng.next_u64();
        debug_assert!({
            prob.iter()
                .fold(0u64, |acc, next| acc.checked_add(*next).unwrap());
            true
        });

        prob.into_iter().enumerate().find_fold(0, |acc, (idx, p)| {
            if roll < p + acc {
                ControlFlow::Break(Self::variants()[idx])
            } else {
                ControlFlow::Continue(p + acc)
            }
        })
    }
}

events!(Genome[NewConnection, BisectConnection, MutateConnection, MutateNode]);
events!(Connection[Disable, MutateParam]);
