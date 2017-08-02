# Change Log

## [0.4.0] - 2016-09-07

* Made all traits use associated types for their output type parameters. This
  simplifies usage of the traits and is consistent with the concept of
  associated types used as output type parameters (not input type parameters) as
  described in [the associated type
  RFC](https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md).
* Implemented `check_info!` macro to check all LAPACK calls.
* Implemented error handling with [error_chain](https://crates.io/crates/error-chain).

## [0.3.0] - 2016-09-06

* Documentation is hosted at https://docs.rs/nalgebra-lapack/
* Updated `nalgebra` to 0.10.
* Rename traits `HasSVD` to `SVD` and `HasEigensystem` to `Eigensystem`.
* Added `Solve` trait for solving a linear matrix equation.
* Added `Inverse` for computing the multiplicative inverse of a matrix.
* Added `Cholesky` for decomposing a positive-definite matrix.
* The `Eigensystem` and `SVD` traits are now generic over types. The
  associated types have been removed.
