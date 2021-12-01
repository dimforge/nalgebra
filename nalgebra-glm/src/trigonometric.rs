use na;

use crate::aliases::TVec;
use crate::RealNumber;

/// Component-wise arc-cosinus.
pub fn acos<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> TVec<T, D> {
    x.map(|e| e.acos())
}

/// Component-wise hyperbolic arc-cosinus.
pub fn acosh<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> TVec<T, D> {
    x.map(|e| e.acosh())
}

/// Component-wise arc-sinus.
pub fn asin<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> TVec<T, D> {
    x.map(|e| e.asin())
}

/// Component-wise hyperbolic arc-sinus.
pub fn asinh<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> TVec<T, D> {
    x.map(|e| e.asinh())
}

/// Component-wise arc-tangent of `y / x`.
pub fn atan2<T: RealNumber, const D: usize>(y: &TVec<T, D>, x: &TVec<T, D>) -> TVec<T, D> {
    y.zip_map(x, |y, x| y.atan2(x))
}

/// Component-wise arc-tangent.
pub fn atan<T: RealNumber, const D: usize>(y_over_x: &TVec<T, D>) -> TVec<T, D> {
    y_over_x.map(|e| e.atan())
}

/// Component-wise hyperbolic arc-tangent.
pub fn atanh<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> TVec<T, D> {
    x.map(|e| e.atanh())
}

/// Component-wise cosinus.
pub fn cos<T: RealNumber, const D: usize>(angle: &TVec<T, D>) -> TVec<T, D> {
    angle.map(|e| e.cos())
}

/// Component-wise hyperbolic cosinus.
pub fn cosh<T: RealNumber, const D: usize>(angle: &TVec<T, D>) -> TVec<T, D> {
    angle.map(|e| e.cosh())
}

/// Component-wise conversion from radians to degrees.
pub fn degrees<T: RealNumber, const D: usize>(radians: &TVec<T, D>) -> TVec<T, D> {
    radians.map(|e| e * na::convert(180.0) / T::pi())
}

/// Component-wise conversion from degrees to radians.
pub fn radians<T: RealNumber, const D: usize>(degrees: &TVec<T, D>) -> TVec<T, D> {
    degrees.map(|e| e * T::pi() / na::convert(180.0))
}

/// Component-wise sinus.
pub fn sin<T: RealNumber, const D: usize>(angle: &TVec<T, D>) -> TVec<T, D> {
    angle.map(|e| e.sin())
}

/// Component-wise hyperbolic sinus.
pub fn sinh<T: RealNumber, const D: usize>(angle: &TVec<T, D>) -> TVec<T, D> {
    angle.map(|e| e.sinh())
}

/// Component-wise tangent.
pub fn tan<T: RealNumber, const D: usize>(angle: &TVec<T, D>) -> TVec<T, D> {
    angle.map(|e| e.tan())
}

/// Component-wise hyperbolic tangent.
pub fn tanh<T: RealNumber, const D: usize>(angle: &TVec<T, D>) -> TVec<T, D> {
    angle.map(|e| e.tanh())
}
