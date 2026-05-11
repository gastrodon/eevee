#!/bin/sh

get_all_benchmarks() {
        toml2json < Cargo.toml | jq -r '.bench[]? | .name'
}

run_bench() {
        bench="$1"
        shift
        svg="flamegraph-$bench-$(git branch --show-current)-$(git rev-parse --short HEAD).svg"
        CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph \
                --features smol_bench,serialize_json -o "$svg" \
                --bench "$bench" \
                -- --bench \
                "$@" \
                && [ -n "$DISPLAY" ] && firefox "$svg"
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
                run_bench "$bench" "$@"
        done
        exit 0
fi

shift
run_bench "$target" "$@"
