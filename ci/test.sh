#! /bin/bash

if [ -z NO_STD ]; then
    cargo test --verbose ${CARGO_FEATURES};

    if [ -z CARGO_FEATURES ]; then
        export CARGO_FEATURE_SYSTEM_NETLIB=1 CARGO_FEATURE_EXCLUDE_LAPACKE=1 CARGO_FEATURE_EXCLUDE_CBLAS=1
        cd nalgebra-lapack; cargo test --verbose;
    fi
fi