use std::fmt;
use num_complex::Complex;

use alga::general::Real;
use core::{Unit, SquareMatrix, Vector1};
use core::dimension::{U2, U3};
use core::allocator::{OwnedAllocator, Allocator};
use core::storage::OwnedStorage;
use geometry::{RotationBase, OwnedRotation};

/// A complex number with a norm equal to 1.
pub type UnitComplex<N> = Unit<Complex<N>>;

impl<N: Real> UnitComplex<N> {
    /// The rotation angle in `]-pi; pi]` of this unit complex number.
    #[inline]
    pub fn angle(&self) -> N {
        self.complex().im.atan2(self.complex().re)
    }

    /// The rotation angle returned as a 1-dimensional vector.
    #[inline]
    pub fn scaled_axis(&self) -> Vector1<N> {
        Vector1::new(self.angle())
    }

    /// The underlying complex number.
    ///
    /// Same as `self.as_ref()`.
    #[inline]
    pub fn complex(&self) -> &Complex<N> {
        self.as_ref()
    }

    /// Compute the conjugate of this unit complex number.
    #[inline]
    pub fn conjugate(&self) -> Self {
        UnitComplex::new_unchecked(self.as_ref().conj())
    }

    /// Inverts this complex number if it is not zero.
    #[inline]
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    #[inline]
    pub fn angle_to(&self, other: &Self) -> N {
        let delta = self.rotation_to(other);
        delta.angle()
    }

    /// The unit complex number needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    #[inline]
    pub fn rotation_to(&self, other: &Self) -> Self {
        other / self
    }

    /// Compute in-place the conjugate of this unit complex number.
    #[inline]
    pub fn conjugate_mut(&mut self) {
        let me = self.as_mut_unchecked();
        me.im = -me.im;
    }

    /// Inverts in-place this unit complex number.
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.conjugate_mut()
    }

    /// Raise this unit complex number to a given floating power.
    ///
    /// This returns the unit complex number that identifies a rotation angle equal to
    /// `self.angle() × n`.
    #[inline]
    pub fn powf(&self, n: N) -> Self {
        Self::from_angle(self.angle() * n)
    }

    /// Builds the rotation matrix corresponding to this unit complex number.
    #[inline]
    pub fn to_rotation_matrix<S>(&self) -> RotationBase<N, U2, S>
        where S: OwnedStorage<N, U2, U2>,
              S::Alloc: OwnedAllocator<N, U2, U2, S> {
        let r = self.complex().re;
        let i = self.complex().im;

        RotationBase::from_matrix_unchecked(
            SquareMatrix::<_, U2, _>::new(
                r, -i,
                i,  r
            )
        )
    }

    /// Converts this unit complex number into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous<S>(&self) -> SquareMatrix<N, U3, S>
        where S: OwnedStorage<N, U3, U3>,
              S::Alloc: OwnedAllocator<N, U3, U3, S> +
                        Allocator<N, U2, U2> {
        let r: OwnedRotation<N, U2, S::Alloc> = self.to_rotation_matrix();
        r.to_homogeneous()
    }
}

impl<N: Real + fmt::Display> fmt::Display for UnitComplex<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "UnitComplex angle: {}", self.angle())
    }
}
