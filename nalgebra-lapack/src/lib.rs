//! # nalgebra-lapack
//! 
//! Rust library for linear algebra using nalgebra and LAPACK.
//! 
//! ## Documentation
//! 
//! Documentation is available [here](https://docs.rs/nalgebra-lapack/).
//! 
//! ## License
//! 
//! MIT
//! 
//! ## Cargo features to select lapack provider
//! 
//! Like the [lapack crate](https://crates.io/crates/lapack) from which this
//! behavior is inherited, nalgebra-lapack uses [cargo
//! features](http://doc.crates.io/manifest.html#the-[features]-section) to select
//! which lapack provider (or implementation) is used. Command line arguments to
//! cargo are the easiest way to do this, and the best provider depends on your
//! particular system. In some cases, the providers can be further tuned with
//! environment variables.
//! 
//! Below are given examples of how to invoke `cargo build` on two different systems
//! using two different providers. The `--no-default-features --features "provider"`
//! arguments will be consistent for other `cargo` commands.
//! 
//! ### Ubuntu
//! 
//! As tested on Ubuntu 12.04, do this to build the lapack package against
//! the system installation of netlib without LAPACKE (note the E) or
//! CBLAS:
//! 
//! ```.ignore
//! sudo apt-get install gfortran libblas3gf liblapack3gf
//! export CARGO_FEATURE_SYSTEM_NETLIB=1
//! export CARGO_FEATURE_EXCLUDE_LAPACKE=1
//! export CARGO_FEATURE_EXCLUDE_CBLAS=1
//! 
//! export CARGO_FEATURES='--no-default-features --features netlib'
//! cargo build ${CARGO_FEATURES}
//! ```
//! 
//! ### Mac OS X
//! 
//! On Mac OS X, do this to use Apple's Accelerate framework:
//! 
//! ```.ignore
//! export CARGO_FEATURES='--no-default-features --features accelerate'
//! cargo build ${CARGO_FEATURES}
//! ```
//! 
//! [version-img]: https://img.shields.io/crates/v/nalgebra-lapack.svg
//! [version-url]: https://crates.io/crates/nalgebra-lapack
//! [status-img]: https://travis-ci.org/strawlab/nalgebra-lapack.svg?branch=master
//! [status-url]: https://travis-ci.org/strawlab/nalgebra-lapack
//! [doc-img]: https://docs.rs/nalgebra-lapack/badge.svg
//! [doc-url]: https://docs.rs/nalgebra-lapack/
//! 
//! ## Contributors
//! This integration of LAPACK on nalgebra was
//! [initiated](https://github.com/strawlab/nalgebra-lapack) by Andrew Straw. It
//! then became officially supported and integrated to the main nalgebra
//! repository.

#![deny(non_camel_case_types)]
#![deny(unused_parens)]
#![deny(non_upper_case_globals)]
#![deny(unused_qualifications)]
#![deny(unused_results)]
#![deny(missing_docs)]
#![doc(html_root_url = "http://nalgebra.org/rustdoc")]

extern crate num_traits as num;
extern crate num_complex;
extern crate lapack;
extern crate alga;
extern crate nalgebra as na;

mod lapack_check;
mod svd;
mod eigen;
mod symmetric_eigen;
mod cholesky;
mod lu;
mod qr;
mod hessenberg;
mod schur;

use num_complex::Complex;

pub use self::svd::SVD;
pub use self::cholesky::{Cholesky, CholeskyScalar};
pub use self::lu::{LU, LUScalar};
pub use self::eigen::Eigen;
pub use self::symmetric_eigen::SymmetricEigen;
pub use self::qr::QR;
pub use self::hessenberg::Hessenberg;
pub use self::schur::RealSchur;


trait ComplexHelper {
    type RealPart;

    fn real_part(self) -> Self::RealPart;
}

impl ComplexHelper for f32 {
    type RealPart = f32;

    #[inline]
    fn real_part(self) -> Self::RealPart {
        self
    }
}

impl ComplexHelper for f64 {
    type RealPart = f64;

    #[inline]
    fn real_part(self) -> Self::RealPart {
        self
    }
}

impl ComplexHelper for Complex<f32> {
    type RealPart = f32;

    #[inline]
    fn real_part(self) -> Self::RealPart {
        self.re
    }
}

impl ComplexHelper for Complex<f64> {
    type RealPart = f64;

    #[inline]
    fn real_part(self) -> Self::RealPart {
        self.re
    }
}

unsafe fn uninitialized_vec<T: Copy>(n: usize) -> Vec<T> {
    let mut res = Vec::new();
    res.reserve_exact(n);
    res.set_len(n);
    res
}
