#!/bin/sh

target="$1"
if [ -z "$target" ]; then
        echo "Target is empty. Exiting."
        exit 1
fi

shift
CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph \
        --features smol_bench -o "flamegraph-$target-$(git branch --show-current)-$(git rev-parse --short HEAD).svg" \
        --bench $target \
        -- --bench \
        $@ \
        && firefox "flamegraph-$target-$(git branch --show-current)-$(git rev-parse --short HEAD).svg"
