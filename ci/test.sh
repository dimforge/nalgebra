#! /bin/bash

set -ev

if [ -z "$NO_STD" ]; then
    if [ -z "$LAPACK" ]; then
        cargo test --verbose;
        cargo test --verbose "arbitrary";
        cargo test --verbose "debug arbitrary mint serde-serialize abomonation-serialize";
        cd nalgebra-glm; cargo test --verbose;
    else
        cd nalgebra-lapack; cargo test --verbose;
    fi
fi