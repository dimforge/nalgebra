#! /bin/bash

if [ -z NO_STD ]; then
    cargo build --verbose --features "${CARGO_FEATURES}";
else
    cargo install xargo;
    xargo build --verbose --no-default-features --target=x86_64-unknown-linux-gnu --features "${CARGO_FEATURES}";
fi