#!/bin/sh

get_all_benchmarks() {
        toml2json < Cargo.toml | jq -r '.bench[]? | .name'
}

target="$1"
if [ -z "$target" ]; then
        echo "Available benchmarks:"
        get_all_benchmarks | sed 's/^/  /'
        exit 0
fi

if [ "$target" = "--all" ]; then
        shift
        for bench in $(get_all_benchmarks); do
                nix develop --command cargo bench --bench "$bench" "$@"
        done
        exit 0
fi

shift
CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph \
        --features smol_bench -o "flamegraph-$target-$(git branch --show-current)-$(git rev-parse --short HEAD).svg" \
        --bench $target \
        -- --bench \
        $@ \
        && firefox "flamegraph-$target-$(git branch --show-current)-$(git rev-parse --short HEAD).svg"
