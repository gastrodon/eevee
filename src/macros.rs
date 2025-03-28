#[macro_export]
macro_rules! new_t {
    ($t:ty, $($k:ident = $v:expr),+ $(,)?) => {{
        let mut c = <$t>::default();
        $(c.$k = $v;)+
        c
    }};
    ($($k:ident = $v:expr),+ $(,)?) => {new_t!(T, $($k = $v,)+)};
}

#[macro_export]
macro_rules! test_t {
  ( #[should_panic(expected = $panic_msg:literal)]
    $name:ident[T: $($impl:ty)|*]() $body:tt ) => {$(
      ::paste::paste! {
          #[test]
          #[should_panic(expected = $panic_msg)]
          fn [<$name _ $impl:snake>]() {
            type T=$impl;
            $body
          }
      }
  )+};
  ($name:ident[T: $($impl:ty)|*]() $body:tt ) => {$(
      ::paste::paste! {
          #[test]
          fn [<test_ $name _ $impl:snake>]() {
            type T=$impl;
            $body
          }
      }
  )+};
}
