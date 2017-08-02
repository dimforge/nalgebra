
#![deny(non_camel_case_types)]
#![deny(unused_parens)]
#![deny(non_upper_case_globals)]
#![deny(unused_qualifications)]
#![deny(unused_results)]
#![warn(missing_docs)]
#![doc(html_root_url = "http://nalgebra.org/rustdoc")]

extern crate num_traits as num;
extern crate num_complex;
extern crate lapack;
extern crate alga;
extern crate nalgebra as na;

mod lapack_check;
mod svd;
mod eigen;
mod cholesky;
mod lu;
mod qr;
mod hessenberg;

use num_complex::Complex;

pub use self::svd::SVD;
pub use self::cholesky::{Cholesky, CholeskyScalar};
pub use self::lu::{LU, LUScalar};
pub use self::eigen::RealEigensystem;
pub use self::qr::QR;
pub use self::hessenberg::Hessenberg;


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
