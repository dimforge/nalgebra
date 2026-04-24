# nalgebra-lapack [![Version][version-img]][version-url] [![Doc][doc-img]][doc-url]

Rust library for linear algebra using nalgebra and LAPACK.

## Documentation

Documentation is available [here](https://docs.rs/nalgebra-lapack/).

## License

MIT

## Cargo features to select LAPACK provider

Like the [lapack crate](https://crates.io/crates/lapack) from which this
behavior is inherited, nalgebra-lapack uses [cargo
features](https://doc.crates.io/manifest.html#the-[features]-section) to select
which LAPACK provider (or implementation) is used. Command line arguments to
cargo are the easiest way to do this, and the best provider depends on your
particular system. In some cases, the providers can be further tuned with
environment variables.

Below are given examples of how to invoke `cargo build` on two different systems
using two different providers. The `--no-default-features --features "lapack-*"`
arguments will be consistent for other `cargo` commands.

### Ubuntu

As tested on Ubuntu 24.04, do this to build the LAPACK package against
the system installation of netlib without LAPACKE (note the E) or
CBLAS:

    sudo apt-get install gfortran libblas-dev liblapack-dev
    export CARGO_FEATURE_SYSTEM_NETLIB=1
    export CARGO_FEATURE_EXCLUDE_LAPACKE=1
    export CARGO_FEATURE_EXCLUDE_CBLAS=1

    export CARGO_FEATURES="--no-default-features --features lapack-netlib"
    cargo build ${CARGO_FEATURES}

### macOS

On macOS, do this to use Apple's Accelerate framework:

    export CARGO_FEATURES="--no-default-features --features lapack-accelerate"
    cargo build ${CARGO_FEATURES}

[version-img]: https://img.shields.io/crates/v/nalgebra-lapack.svg
[version-url]: https://crates.io/crates/nalgebra-lapack
[doc-img]: https://docs.rs/nalgebra-lapack/badge.svg
[doc-url]: https://docs.rs/nalgebra-lapack/

## Contributors
This integration of LAPACK on nalgebra was
[initiated](https://github.com/strawlab/nalgebra-lapack) by Andrew Straw. It
then became officially supported and integrated to the main nalgebra
repository.
