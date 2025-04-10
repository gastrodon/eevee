# NeatKit

### This project is a WIP!

Nothing really works very well. There's a lot of useful code, and topology search / genome evolution _can_ be done,
but it's slow, inefficient, and often fails completely. Expect frequent changes.

### Overview

NeatKit is a library for leveraging the [NEAT](https://web.archive.org/web/20241209001646/https://ieeexplore.ieee.org/abstract/document/6790655) algorithm to train genomes encoding neural network behavior. Specifically, it aims to generalize the algorithm such that it may be applied to different domains, and maybe in the future applications that don't implement neural networks at all.

### Try it and see

The core iteration loop is that, given a scenario which implements some mechanism by which a genome may be scored with a fitness, NeatKit will try mutate, reproduce, and cull genomes to optimize for that fitness to increase. There exist some experiments aound this in the `examples` folder.

### Building on it

It's written in rust, and uses nightly versions - mostly for incomplete features. Later when I add CUDA support, it will even more rely on nightly.

I use [criterion](https://github.com/bheisler/criterion.rs) for benchmarking, it's recommended that if you run benches, you have `gnuplot` on your system. You can use `./cmp-bench <bench> [branch:-]` to compare a benchmark across two branches, which produces a nice report.

I use [flamegraph](https://github.com/flamegraph-rs/flamegraph) for profiling, it's required that if you run benches with profiling, you have `perf` on your system. You can use `./profile <bench>` to run benchmarks on a pared-down version of any benchmark, and `./cmp-profile <bench> [branch:-]` to compare a profiling across two branches.

### Other things

Thanks to [smol-rs/fastrand](https://github.com/smol-rs/fastrand), I stole the core of their `WyHash` rng implementation.

Thanks to [TLmaK0/rustneat](https://github.com/TLmaK0/rustneat/), This project turned me on to learning about CTRNN's and also I stole the CTRNN matmul code
