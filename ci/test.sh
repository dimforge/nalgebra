#! /bin/bash

set -ev

if [ -z "$NO_STD" ]; then
    if [ -z "$LAPACK" ]; then
        cargo test --verbose --no-default-features --features "std" --tests;
        cargo test --verbose;
        cargo test --verbose --features "arbitrary";
        cargo test --verbose --all-features;
        cd nalgebra-glm; cargo test --verbose;
    else
        cd nalgebra-lapack; cargo test --verbose;
    fi
fi
