use criterion::Criterion;
use eevee::random::{seed_time, seed_pid_time, seed_thread_time, seed_urandom};

macro_rules! bench_seed_fn {
    ($bench:expr, $fn_name:ident) => {
        $bench.bench_function(concat!("seed-", stringify!($fn_name)), |b| {
            b.iter(|| $fn_name());
        });
    };
}

fn bench_seed_functions(bench: &mut Criterion) {
    bench_seed_fn!(bench, seed_time);
    bench_seed_fn!(bench, seed_pid_time);
    bench_seed_fn!(bench, seed_thread_time);
    
    // seed_urandom returns Result, so handle it differently
    bench.bench_function("seed-seed_urandom", |b| {
        b.iter(|| seed_urandom().unwrap());
    });
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(1000)
        .significance_level(0.1);
    #[cfg(feature = "smol_bench")]
    let mut criterion: criterion::Criterion<_> = {
        use core::time::Duration;
        Criterion::default()
            .measurement_time(Duration::from_millis(1))
            .sample_size(10)
            .nresamples(1)
            .without_plots()
            .configure_from_args()
    };
    bench_seed_functions(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
