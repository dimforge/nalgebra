//! Construction of givens rotations.

use alga::general::{Complex, Real};
use num_complex::Complex as NumComplex;

use base::dimension::{Dim, U2};
use base::constraint::{ShapeConstraint, DimEq};
use base::storage::{Storage, StorageMut};
use base::{Vector, Matrix};

use geometry::UnitComplex;

/// A Givens rotation.
pub struct GivensRotation<N: Complex> {
    c: N,
    s: N
}

// XXX: remove this
/// Computes the rotation `R` required such that the `y` component of `R * v` is zero.
///
/// Returns `None` if no rotation is needed (i.e. if `v.y == 0`). Otherwise, this returns the norm
/// of `v` and the rotation `r` such that `R * v = [ |v|, 0.0 ]^t` where `|v|` is the norm of `v`.
pub fn cancel_y<N: Real, S: Storage<N, U2>>(v: &Vector<N, U2, S>) -> Option<(UnitComplex<N>, N)> {
    if !v[1].is_zero() {
        let c = NumComplex::new(v[0], -v[1]);
        Some(UnitComplex::from_complex_and_get(c))
    } else {
        None
    }
}

// XXX: remove this
/// Computes the rotation `R` required such that the `x` component of `R * v` is zero.
///
/// Returns `None` if no rotation is needed (i.e. if `v.x == 0`). Otherwise, this returns the norm
/// of `v` and the rotation `r` such that `R * v = [ 0.0, |v| ]^t` where `|v|` is the norm of `v`.
pub fn cancel_x<N: Real, S: Storage<N, U2>>(v: &Vector<N, U2, S>) -> Option<(UnitComplex<N>, N)> {
    if !v[0].is_zero() {
        let c = NumComplex::new(v[1], v[0]);
        Some(UnitComplex::from_complex_and_get(c))
    } else {
        None
    }
}


// Matrix = UnitComplex * Matrix
impl<N: Complex> GivensRotation<N> {
    /// Initializes a Givens rotation form its non-normalized cosine an sine components.
    pub fn new(c: N, s: N) -> Self {
        let denom = (c.modulus_squared() + s.modulus_squared()).sqrt();
        Self {
            c: c.unscale(denom),
            s: s.unscale(denom)
        }
    }

    /// Initializes a Givens rotation form its non-normalized cosine an sine components.
    pub fn try_new(c: N, s: N, eps: N::Real) -> Option<Self> {
        let denom = (c.modulus_squared() + s.modulus_squared()).sqrt();

        if denom > eps {
            Some(Self {
                c: c.unscale(denom),
                s: s.unscale(denom)
            })
        } else {
            None
        }
    }

    /// Computes the rotation `R` required such that the `y` component of `R * v` is zero.
    ///
    /// Returns `None` if no rotation is needed (i.e. if `v.y == 0`). Otherwise, this returns the norm
    /// of `v` and the rotation `r` such that `R * v = [ |v|, 0.0 ]^t` where `|v|` is the norm of `v`.
    pub fn cancel_y<S: Storage<N, U2>>(v: &Vector<N, U2, S>) -> Option<(Self, N)> {
        if !v[1].is_zero() {
            let (mod0, sign0) = v[0].to_exp();
            let denom = (mod0 * mod0 + v[1].modulus_squared()).sqrt();
            let c = N::from_real(mod0 / denom);
            let s = (sign0 * v[1].conjugate()).unscale(-denom);
            let r = sign0.scale(denom);
            Some((Self { c, s }, r))
        } else {
            None
        }
    }

    /// Computes the rotation `R` required such that the `x` component of `R * v` is zero.
    ///
    /// Returns `None` if no rotation is needed (i.e. if `v.x == 0`). Otherwise, this returns the norm
    /// of `v` and the rotation `r` such that `R * v = [ 0.0, |v| ]^t` where `|v|` is the norm of `v`.
    pub fn cancel_x<S: Storage<N, U2>>(v: &Vector<N, U2, S>) -> Option<(Self, N)> {
        if !v[0].is_zero() {
            let (mod0, sign0) = v[0].to_exp();
            let denom = (mod0 * mod0 + v[1].modulus_squared()).sqrt();
            let c = N::from_real(mod0 / denom);
            let s = (sign0 * v[1].conjugate()).unscale(denom);
            let r = sign0.scale(denom);
            Some((Self { c, s }, r))
        } else {
            None
        }
    }

    /// The cos part of this roration.
    pub fn c(&self) -> N {
        self.c
    }

    /// The sin part of this roration.
    pub fn s(&self) -> N {
        self.s
    }

    /// The inverse of this givens rotation.
    pub fn inverse(&self) -> Self {
        Self { c: self.c, s: -self.s.conjugate() }
    }

    /// Performs the multiplication `rhs = self * rhs` in-place.
    pub fn rotate<R2: Dim, C2: Dim, S2: StorageMut<N, R2, C2>>(
        &self,
        rhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        ShapeConstraint: DimEq<R2, U2>,
    {
        assert_eq!(
            rhs.nrows(),
            2,
            "Unit complex rotation: the input matrix must have exactly two rows."
        );
        let s = self.s;
        let c = self.c;

        for j in 0..rhs.ncols() {
            unsafe {
                let a = *rhs.get_unchecked((0, j));
                let b = *rhs.get_unchecked((1, j));

                *rhs.get_unchecked_mut((0, j)) = c * a - s.conjugate() * b;
                *rhs.get_unchecked_mut((1, j)) = s * a + c.conjugate() * b;
            }
        }
    }

    /// Performs the multiplication `lhs = lhs * self` in-place.
    pub fn rotate_rows<R2: Dim, C2: Dim, S2: StorageMut<N, R2, C2>>(
        &self,
        lhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        ShapeConstraint: DimEq<C2, U2>,
    {
        assert_eq!(
            lhs.ncols(),
            2,
            "Unit complex rotation: the input matrix must have exactly two columns."
        );
        let s = self.s;
        let c = self.c;

        // FIXME: can we optimize that to iterate on one column at a time ?
        for j in 0..lhs.nrows() {
            unsafe {
                let a = *lhs.get_unchecked((j, 0));
                let b = *lhs.get_unchecked((j, 1));

                *lhs.get_unchecked_mut((j, 0)) = c * a + s * b;
                *lhs.get_unchecked_mut((j, 1)) = -s.conjugate() * a + c.conjugate() * b;
            }
        }
    }
}

