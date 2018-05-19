//! Construction of givens rotations.

use alga::general::Real;
use num_complex::Complex;

use base::dimension::U2;
use base::storage::Storage;
use base::Vector;

use geometry::UnitComplex;

/// Computes the rotation `R` required such that the `y` component of `R * v` is zero.
///
/// Returns `None` if no rotation is needed (i.e. if `v.y == 0`). Otherwise, this returns the norm
/// of `v` and the rotation `r` such that `R * v = [ |v|, 0.0 ]^t` where `|v|` is the norm of `v`.
pub fn cancel_y<N: Real, S: Storage<N, U2>>(v: &Vector<N, U2, S>) -> Option<(UnitComplex<N>, N)> {
    if !v[1].is_zero() {
        let c = Complex::new(v[0], -v[1]);
        Some(UnitComplex::from_complex_and_get(c))
    } else {
        None
    }
}

/// Computes the rotation `R` required such that the `x` component of `R * v` is zero.
///
/// Returns `None` if no rotation is needed (i.e. if `v.x == 0`). Otherwise, this returns the norm
/// of `v` and the rotation `r` such that `R * v = [ 0.0, |v| ]^t` where `|v|` is the norm of `v`.
pub fn cancel_x<N: Real, S: Storage<N, U2>>(v: &Vector<N, U2, S>) -> Option<(UnitComplex<N>, N)> {
    if !v[0].is_zero() {
        let c = Complex::new(v[1], v[0]);
        Some(UnitComplex::from_complex_and_get(c))
    } else {
        None
    }
}
