#! /bin/bash

if [ -z "$NO_STD" ]; then
    cargo build --verbose;
    cargo build --verbose --features "arbitrary";
    cargo build --verbose --features "mint";
    cargo build --verbose --features "alloc";
    cargo build --verbose --features "serde-serialize";
    cargo build --verbose --features "abomonation-serialize";
    cargo build --verbose --features "debug";
    cargo build --verbose --features "debug arbitrary mint serde-serialize abomonation-serialize";
else
    rustup component add rust-src
    cargo install xargo;
    xargo build --verbose --no-default-features --target=x86_64-unknown-linux-gnu --features "${CARGO_FEATURES}";
fi