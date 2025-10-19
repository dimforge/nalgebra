//! # nalgebra-lapack
//!
//! Rust library for linear algebra using nalgebra and LAPACK.
//!
//! ## Selecting a LAPACK Backend
//!
//! This crate uses cargo [cargo features](https://doc.crates.io/manifest.html#the-[features]-section)
//! to select which lapack provider (or implementation) is used.
//!
//! ### Default LAPACK Backend and Performance
//!
//! By default, the [`netlib-src`](https://crates.io/crates/netlib-src) crate
//! is used as a LAPACK source, which bundles netlib and works out of the box,
//! given an existing FOTRAN compiler on your system. That makes this choice
//! practical, but it's typically not the best performing backend.
//!
//! ### Lapack Backends
//!
//! LAPACK backends other than `netlib` typically assume the libraries or
//! frameworks are present on your system. See the respective vendors how to
//! install them. Backends are selected using one of the `lapack-*` feature flags:
//!
//! * `lapack-netlib`: use the bundled [Netlib](http://www.netlib.org/) reference
//!   implementation. This feature is enabled by default.
//! * `lapack-openblas`: Use LAPACK provided via [OpenBLAS](http://www.openmathlib.org/OpenBLAS/).
//! * `lapack-accelerate`: Use Apple's [Accelerate](https://developer.apple.com/documentation/accelerate)
//!   framework.
//! * `lapack-mkl`: alias for `lapack-mkl-static-seq`. A useful default for [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
//! * `lapack-mkl-static-seq`: statically link the _sequential_ version of MKL.
//! * `lapack-mkl-static-par`: statically link the _parallel_ version of MKL.
//! * `lapack-mkl-dynamic-seq`: dynamically link the sequential version of MKL.
//! * `lapack-mkl-dynamic-par`: dynamically link the parallel version of MKL.
//! * `lapack-custom`: Use a custom lapack backend whose functions must be
//!   available at linktime. It is your responsibility to make sure those are
//!   ABI compatible to the function signatures in the [lapack](https://crates.io/crates/lapack)
//!   crate.
//!
//! Note that **exactly one** of these features must be selected
//! and that `lapack-netlib` is selected by default, which means **you have
//! to disable default features**, when explicitly specifying a lapack backend.
//!
//! ```toml
//! nalgebra-lapack = {version = *, default-features = false, features = ["lapack-*"]}
//! ```
//!
//! [version-img]: https://img.shields.io/crates/v/nalgebra-lapack.svg
//! [version-url]: https://crates.io/crates/nalgebra-lapack
//! [doc-img]: https://docs.rs/nalgebra-lapack/badge.svg
//! [doc-url]: https://docs.rs/nalgebra-lapack/
//!
//! ## Performance
//!
//! As always, there's only one way to find out if nalgebra-lapack brings
//! a performance benefit to your project, which is _measuring_. The same
//! goes for deciding which LAPACK backend to use, if you have multiple available
//! options.
//!
//! ## License
//!
//! MIT
//!
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
#![doc(
    html_favicon_url = "https://nalgebra.rs/img/favicon.ico",
    html_root_url = "https://nalgebra.rs/rustdoc"
)]

// a utility macro that makes sure that exactly one of the lapack backend
// features is selected. It provides helpful error messages otherwise.
macro_rules! enforce_exactly_one_feature_selected {
    ($($feat:literal),+ $(,)?) => {
        const _ENABLED_FEATURE_COUNT : usize = $(if cfg!(feature = $feat) {1} else {0} +)* 0;
        const _ASSERT_EXACTLY_ONE : () = if _ENABLED_FEATURE_COUNT > 1 {
            panic!("Multiple `lapack-*` features selected! Make sure to set `default-features = false` when including nalgebra-lapack with explicit `lapack-*` features.");
        } else if _ENABLED_FEATURE_COUNT == 0 {
            panic!("Select at least one `lapack-*` feature! To use a custom backend at link-time, specify `lapack-custom` without default features.");
        } else {};
    };
}

// If you hit a compile error here, make sure to specify exactly one of the
// available `lapack-*` features in this crate.
//
// # Troubleshooting
//
// ## Multiple Features Selected!
//
// By default, the `lapack-netlib`
// backend is specified, so to select another backend, you have disable
// the default-features of this crate, when including this crate.
//
// ## Select At Least One Feature!
//
// If you want to use a custom backend at link-time, turn off default features
// and use `lapack-custom`.
enforce_exactly_one_feature_selected!(
    "lapack-openblas",
    "lapack-netlib",
    "lapack-accelerate",
    "lapack-mkl-static-seq",
    "lapack-mkl-static-par",
    "lapack-mkl-dynamic-seq",
    "lapack-mkl-dynamic-par",
    "lapack-custom"
);

extern crate lapack;
#[cfg(not(feature = "lapack-custom"))]
extern crate lapack_src;

extern crate nalgebra as na;
extern crate num_traits as num;

mod lapack_check;

mod cholesky;
/// column-pivoted QR decomposition of a rectangular (or square) matrix
pub mod colpiv_qr;
mod eigen;
mod generalized_eigenvalues;
mod hessenberg;
mod lu;
/// QR decomposition of a rectangular (or square) matrix
pub mod qr;
mod qz;
mod schur;
mod svd;
mod symmetric_eigen;

/// internal utility module that contains functionality that is useful for
/// both column-pivoted and non-pivoted qr decomposition
mod qr_util;

use num_complex::Complex;

/// utility module that defines some common terms that lapack uses
mod lapack_terminology;

/// utility module for lapack error codes and error checking
mod lapack_error;

pub use lapack_error::LapackErrorCode;
pub use lapack_terminology::DiagonalKind;
pub use lapack_terminology::Side;
pub use lapack_terminology::Transposition;
pub use lapack_terminology::TriangularStructure;

pub use self::cholesky::{Cholesky, CholeskyScalar};
pub use self::colpiv_qr::ColPivQR;
pub use self::eigen::Eigen;
pub use self::generalized_eigenvalues::GeneralizedEigen;
pub use self::hessenberg::Hessenberg;
pub use self::lu::{LU, LUScalar};
pub use self::qr::QR;
pub use self::qr::abstraction::QrDecomposition;
pub use self::qz::QZ;
pub use self::schur::Schur;
pub use self::svd::SVD;
pub use self::symmetric_eigen::SymmetricEigen;

mod sealed;

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
