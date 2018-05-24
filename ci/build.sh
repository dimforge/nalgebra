#! /bin/bash

set -ev

if [ -z "$NO_STD" ]; then
    if [ -z "$LAPACK" ]; then
        cargo build --verbose -p nalgebra;
        cargo build --verbose -p nalgebra --features "arbitrary";
        cargo build --verbose -p nalgebra --features "mint";
        cargo build --verbose -p nalgebra --features "alloc";
        cargo build --verbose -p nalgebra --features "serde-serialize";
        cargo build --verbose -p nalgebra --features "abomonation-serialize";
        cargo build --verbose -p nalgebra --features "debug";
        cargo build --verbose -p nalgebra --features "debug arbitrary mint serde-serialize abomonation-serialize";
    else
        cargo build -p nalgebra-lapack;
    fi
else
    if [ "$CARGO_FEATURES" == "alloc" ]; then
        cat << EOF > Xargo.toml
[target.x86_64-unknown-linux-gnu.dependencies]
alloc = {}
EOF
    fi
    rustup component add rust-src
    cargo install xargo
    xargo build --verbose --no-default-features --target=x86_64-unknown-linux-gnu --features "${CARGO_FEATURES}";
fi