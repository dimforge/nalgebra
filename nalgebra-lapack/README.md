# nalgebra-lapack [![Version][version-img]][version-url] [![Status][status-img]][status-url] [![Doc][doc-img]][doc-url]

Rust library for linear algebra using nalgebra and LAPACK

## Documentation

Documentation is available [here](https://docs.rs/nalgebra-lapack/).

## License

MIT

## Cargo features to select lapack provider

Like the [lapack crate](https://crates.io/crates/lapack) from which this
behavior is inherited, nalgebra-lapack uses [cargo
features](http://doc.crates.io/manifest.html#the-[features]-section) to select
which lapack provider (or implementation) is used. Command line arguments to
cargo are the easiest way to do this, and the best provider depends on your
particular system. In some cases, the providers can be further tuned with
environment variables.

Below are given examples of how to invoke `cargo build` on two different systems
using two different providers. The `--no-default-features --features "provider"`
arguments will be consistent for other `cargo` commands.

### Ubuntu

As tested on Ubuntu 12.04, do this to build the lapack package against
the system installation of netlib without LAPACKE (note the E) or
CBLAS:

    sudo apt-get install gfortran libblas3gf liblapack3gf
    export CARGO_FEATURE_SYSTEM_NETLIB=1
    export CARGO_FEATURE_EXCLUDE_LAPACKE=1
    export CARGO_FEATURE_EXCLUDE_CBLAS=1

    export CARGO_FEATURES='--no-default-features --features netlib'
    cargo build ${CARGO_FEATURES}

### Mac OS X

On Mac OS X, do this to use Apple's Accelerate framework:

    export CARGO_FEATURES='--no-default-features --features accelerate'
    cargo build ${CARGO_FEATURES}

[version-img]: https://img.shields.io/crates/v/nalgebra-lapack.svg
[version-url]: https://crates.io/crates/nalgebra-lapack
[status-img]: https://travis-ci.org/strawlab/nalgebra-lapack.svg?branch=master
[status-url]: https://travis-ci.org/strawlab/nalgebra-lapack
[doc-img]: https://docs.rs/nalgebra-lapack/badge.svg
[doc-url]: https://docs.rs/nalgebra-lapack/
