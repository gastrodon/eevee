use core::cmp::min;
use rand::RngCore;
use std::{
    fs::File,
    io::{self, Read},
    ops::ControlFlow,
};

pub const fn percent(x: u64) -> u64 {
    x * (u64::MAX / 100)
}

pub fn seed_urandom() -> io::Result<u64> {
    let mut file = File::open("/dev/urandom")?;
    let mut buffer = [0u8; 8];
    file.read_exact(&mut buffer)?;
    Ok(u64::from_le_bytes([
        buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
    ]))
}

pub fn default_rng() -> impl RngCore {
    WyRng::seeded(seed_urandom().unwrap())
}

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

trait FindFold<T> {
    fn find_fold<U, R>(&mut self, init: R, op: impl Fn(R, T) -> ControlFlow<U, R>) -> Option<U>;
}

impl<T, I: Iterator<Item = T> + Sized> FindFold<T> for I {
    fn find_fold<U, R>(
        &mut self,
        mut init: R,
        op: impl Fn(R, T) -> ControlFlow<U, R>,
    ) -> Option<U> {
        loop {
            match self.next() {
                Some(v) => match op(init, v) {
                    ControlFlow::Continue(n) => init = n,
                    ControlFlow::Break(res) => break Some(res),
                },
                None => break None,
            }
        }
    }
}

#[macro_export]
macro_rules! count {
    ($_:ident) => {
        1
    };
    ($_:ident, $($remain:ident),+) => {
        1+$crate::count!($($remain),+)
    };
}

#[macro_export]
macro_rules! iota {
    (@inner $t:ty, $name:ident, $value:expr, $($rest:ident, $new_value:expr),*) => {
        const $name: $t = $value;
        $crate::iota!(@inner $t, $($rest, $value + 1),*);
    };
    (@inner $t:ty, $name:ident, $value:expr) => {
        const $name: $t = $value;
    };
    ($t:ty, $($name:ident,)* $(,)?) => {
        $crate::iota!(@inner $t, $($name, 0),*);
    };
}

// TODO not pub structs
#[macro_export]
macro_rules! events {
    ($scope:ident[$($evt:ident),+]) => {
        ::paste::paste! {
            #[derive(Debug, Clone, Copy)]
            pub enum [<$scope Event>] {
                $($evt,)*
            }

            impl $crate::random::EventKind for [<$scope Event>] {
                const COUNT: usize = $crate::count!($($evt),+);

                fn variants() -> [Self; Self::COUNT] {
                    [$(Self::$evt),*]
                }

                fn idx(&self) -> usize {
                    $crate::iota!(usize, $([<$evt:snake:upper _IDX>],)*);
                    match self {
                        $(Self::$evt => [<$evt:snake:upper _IDX>],)*
                    }
                }
            }
        }
    };
}

events!(Genome[NewConnection, BisectConnection, MutateConnection, MutateNode]);
events!(Connection[Disable, MutateParam]);
