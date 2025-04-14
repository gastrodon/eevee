//! Macros for when it's nice to write less code.

/// A macro for instantiating a [Default]-able something, and then assigning some values to it.
#[macro_export]
macro_rules! new_t {
    ($t:ty, $($k:ident = $v:expr),+ $(,)?) => {{
        let mut c = <$t>::default();
        $(c.$k = $v;)+
        c
    }};
    ($($k:ident = $v:expr),+ $(,)?) => {new_t!(T, $($k = $v,)+)};
}

/// A macro for constructing a single test with a number of types.
#[macro_export]
macro_rules! test_t {
    (@panic $name:ident[T: $($impl:ty)|*]() $body:tt ) => {$(
        ::paste::paste! {
            #[test_case]
            fn [<$name _ $impl:snake>]() {
                type T=$impl;
                std::panic::set_hook(Box::new(|_|()));
                let unwound = std::panic::catch_unwind(|| { $body });
                let _ = std::panic::take_hook();
                assert!(unwound.is_err(), "did not panic!");
            }
        }
    )+};
  ($name:ident[T: $($impl:ty)|*]() $body:tt ) => {$(
      ::paste::paste! {
          #[test_case]
          fn [<$name _ $impl:snake>]() {
            type T=$impl;
            $body
          }
      }
  )+};
}

#[macro_export]
macro_rules! assert_f64_approx {
    ($l:expr, $r:expr) => {
        assert!(
            ($l - $r).abs() < f64::EPSILON,
            "assertion failed: {} !~ {}",
            $l,
            $r
        )
    };
    ($l:expr, $r:expr, $msg:expr) => {
        assert!(
            ($l - $r).abs() < f64::EPSILON,
            "assertion failed: {} !~ {}: {}",
            $l,
            $r,
            $msg
        )
    };
}

#[macro_export]
macro_rules! assert_matrix_approx {
    ($a:expr, $b:expr) => {
        assert_eq!($a.len(), $b.len(), "Matrices have different lengths");

        for (i, (l, r)) in $a.iter().zip($b.iter()).enumerate() {
            $crate::assert_f64_approx!(l, r, format!("differs at [{i}]"));
        }
    };
}

#[macro_export]
macro_rules! normalized {
    ($x:expr; $({.$($norm:tt)+})+) => {{
        let mut x = $x;
        $(x.$($norm)*;)+
        x
    }};
}

/// A macro for comparing some values who may differ in ways we don't care about
#[macro_export]
macro_rules! assert_some_normalized {
  ($l:expr, [$($r:expr),*  $(,)?]; $({.$($norm:tt)+})+, $msg: expr) => {{
    let l = $crate::normalized!($l.to_owned(); $({.$($norm)* })+);
    assert!([$($r,)*].into_iter().any(|r| l == $crate::normalized!(r.to_owned(); $({.$($norm)* })+)), "{}", $msg)
  }};
  ($l:expr, [$($r:expr),* $(,)?]; $({.$($norm:tt)+})+) => {$crate::assert_some_normalized!($l, [$($r,)*]; $({.$($norm)* })+, format!("{:?} not in {:?}", $l, [$($r,)*]))};
}

#[macro_export]
macro_rules! mutate_param {
    ([$($evt:ident),+]: [$($prob:expr),+]) => {
        ::paste::paste! {
            fn mutate_param(&mut self, rng: &mut impl rand::RngCore) {
                use $crate::random::EventKind;
                use rand::Rng;
                $crate::events!(Param[$($evt),*]);
                const PARAM_PROBABILITIES: [u64; ParamEvent::COUNT] = [$($prob),*];

                if let Some(evt) = ParamEvent::pick(rng, PARAM_PROBABILITIES) {
                    let replace = rng.next_u64() < Self::PARAM_REPLACE_PROBABILITY;
                    let v: f64 = rng.sample(rand::distr::Uniform::new_inclusive(-3., 3.).expect("distribution of -3. ..= 3. failed"));
                    match evt {
                        $(ParamEvent::[<$evt:camel>] => self.[<$evt:lower>] = if replace {
                            v
                        } else {
                            self.[<$evt:lower>] + ( Self::PARAM_PERTURB_FAC * v )
                        },)*
                    }
                }
            }

            fn param_diff(&self, other: &Self) -> f64 {
                [$((self.[<$evt:lower>] - other.[<$evt:lower>])),*].iter().sum()
            }
        }
    };
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

#[macro_export]
macro_rules! test_data {
    ($p:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/test-data/", $p))
    };
}
