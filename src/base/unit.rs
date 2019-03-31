use approx::RelativeEq;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};
use std::mem;
use std::ops::{Deref, Neg};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::{SubsetOf, ComplexField};
use alga::linear::NormedSpace;

/// A wrapper that ensures the underlying algebraic entity has a unit norm.
///
/// Use `.as_ref()` or `.into_inner()` to obtain the underlying value by-reference or by-move.
#[repr(transparent)]
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
pub struct Unit<T> {
    value: T,
}

#[cfg(feature = "serde-serialize")]
impl<T: Serialize> Serialize for Unit<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        self.value.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Unit<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        T::deserialize(deserializer).map(|x| Unit { value: x })
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<T: Abomonation> Abomonation for Unit<T> {
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.value.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.value.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.value.exhume(bytes)
    }
}

impl<T: NormedSpace> Unit<T> {
    /// Normalize the given value and return it wrapped on a `Unit` structure.
    #[inline]
    pub fn new_normalize(value: T) -> Self {
        Self::new_and_get(value).0
    }

    /// Attempts to normalize the given value and return it wrapped on a `Unit` structure.
    ///
    /// Returns `None` if the norm was smaller or equal to `min_norm`.
    #[inline]
    pub fn try_new(value: T, min_norm: T::RealField) -> Option<Self> {
        Self::try_new_and_get(value, min_norm).map(|res| res.0)
    }

    /// Normalize the given value and return it wrapped on a `Unit` structure and its norm.
    #[inline]
    pub fn new_and_get(mut value: T) -> (Self, T::RealField) {
        let n = value.normalize_mut();

        (Unit { value: value }, n)
    }

    /// Normalize the given value and return it wrapped on a `Unit` structure and its norm.
    ///
    /// Returns `None` if the norm was smaller or equal to `min_norm`.
    #[inline]
    pub fn try_new_and_get(mut value: T, min_norm: T::RealField) -> Option<(Self, T::RealField)> {
        if let Some(n) = value.try_normalize_mut(min_norm) {
            Some((Unit { value: value }, n))
        } else {
            None
        }
    }

    /// Normalizes this value again. This is useful when repeated computations
    /// might cause a drift in the norm because of float inaccuracies.
    ///
    /// Returns the norm before re-normalization. See `.renormalize_fast` for a faster alternative
    /// that may be slightly less accurate if `self` drifted significantly from having a unit length.
    #[inline]
    pub fn renormalize(&mut self) -> T::RealField {
        self.value.normalize_mut()
    }

    /// Normalizes this value again using a first-order Taylor approximation.
    /// This is useful when repeated computations might cause a drift in the norm
    /// because of float inaccuracies.
    #[inline]
    pub fn renormalize_fast(&mut self) {
        let sq_norm = self.value.norm_squared();
        let _3: T::RealField = crate::convert(3.0);
        let _0_5: T::RealField = crate::convert(0.5);
        self.value *= T::ComplexField::from_real(_0_5 * (_3 - sq_norm));
    }
}

impl<T> Unit<T> {
    /// Wraps the given value, assuming it is already normalized.
    #[inline]
    pub fn new_unchecked(value: T) -> Self {
        Unit { value: value }
    }

    /// Wraps the given reference, assuming it is already normalized.
    #[inline]
    pub fn from_ref_unchecked<'a>(value: &'a T) -> &'a Self {
        unsafe { mem::transmute(value) }
    }

    /// Retrieves the underlying value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Retrieves the underlying value.
    /// Deprecated: use [Unit::into_inner] instead.
    #[deprecated(note="use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> T {
        self.value
    }

    /// Returns a mutable reference to the underlying value. This is `_unchecked` because modifying
    /// the underlying value in such a way that it no longer has unit length may lead to unexpected
    /// results.
    #[inline]
    pub fn as_mut_unchecked(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T> AsRef<T> for Unit<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

/*
 *
 * Conversions.
 *
 */
impl<T: NormedSpace> SubsetOf<T> for Unit<T>
where T::Field: RelativeEq
{
    #[inline]
    fn to_superset(&self) -> T {
        self.clone().into_inner()
    }

    #[inline]
    fn is_in_subset(value: &T) -> bool {
        relative_eq!(value.norm_squared(), crate::one())
    }

    #[inline]
    unsafe fn from_superset_unchecked(value: &T) -> Self {
        Unit::new_normalize(value.clone()) // We still need to re-normalize because the condition is inexact.
    }
}

// impl<T: RelativeEq> RelativeEq for Unit<T> {
//     type Epsilon = T::Epsilon;
//
//     #[inline]
//     fn default_epsilon() -> Self::Epsilon {
//         T::default_epsilon()
//     }
//
//     #[inline]
//     fn default_max_relative() -> Self::Epsilon {
//         T::default_max_relative()
//     }
//
//     #[inline]
//     fn default_max_ulps() -> u32 {
//         T::default_max_ulps()
//     }
//
//     #[inline]
//     fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
//         self.value.relative_eq(&other.value, epsilon, max_relative)
//     }
//
//     #[inline]
//     fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
//         self.value.ulps_eq(&other.value, epsilon, max_ulps)
//     }
// }

// FIXME:re-enable this impl when specialization is possible.
// Currently, it is disabled so that we can have a nice output for the `UnitQuaternion` display.
/*
impl<T: fmt::Display> fmt::Display for Unit<T> {
    // XXX: will not always work correctly due to rounding errors.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(f)
    }
}
*/

impl<T: Neg> Neg for Unit<T> {
    type Output = Unit<T::Output>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::new_unchecked(-self.value)
    }
}

impl<T> Deref for Unit<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { mem::transmute(self) }
    }
}
