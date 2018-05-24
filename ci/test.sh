#! /bin/bash

if [ -z "$NO_STD" ]; then
    cargo test --verbose;
    cargo test --verbose "debug arbitrary mint serde-serialize abomonation-serialize";
    cd nalgebra-lapack; cargo test --verbose;
fi