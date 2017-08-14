use num::{Zero, One};
use std::fmt;
use approx::ApproxEq;

#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Serializer, Deserialize, Deserializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::{Real, ClosedNeg};

use core::{Scalar, ColumnVector, OwnedSquareMatrix};
use core::dimension::{DimName, DimNameSum, DimNameAdd, U1};
use core::storage::{Storage, StorageMut, Owned};
use core::allocator::Allocator;

/// A translation with an owned vector storage.
pub type OwnedTranslation<N, D, S> = TranslationBase<N, D, Owned<N, D, U1, <S as Storage<N, D, U1>>::Alloc>>;

/// A translation.
#[repr(C)]
#[derive(Hash, Debug, Clone, Copy)]
pub struct TranslationBase<N: Scalar, D: DimName, S/*: Storage<N, D, U1>*/> {
    /// The translation coordinates, i.e., how much is added to a point's coordinates when it is
    /// translated.
    pub vector: ColumnVector<N, D, S>
}

#[cfg(feature = "serde-serialize")]
impl<N, D, S> Serialize for TranslationBase<N, D, S>
    where N: Scalar,
          D: DimName,
          ColumnVector<N, D, S>: Serialize,
{
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
        where T: Serializer
    {
        self.vector.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de, N, D, S> Deserialize<'de> for TranslationBase<N, D, S>
    where N: Scalar,
          D: DimName,
          ColumnVector<N, D, S>: Deserialize<'de>,
{
    fn deserialize<T>(deserializer: T) -> Result<Self, T::Error>
        where T: Deserializer<'de>
    {
        ColumnVector::deserialize(deserializer).map(|x| TranslationBase { vector: x })
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N, D, S> Abomonation for TranslationBase<N, D, S>
    where N: Scalar,
          D: DimName,
          ColumnVector<N, D, S>: Abomonation
{
    unsafe fn entomb(&self, writer: &mut Vec<u8>) {
        self.vector.entomb(writer)
    }

    unsafe fn embalm(&mut self) {
        self.vector.embalm()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.vector.exhume(bytes)
    }
}

impl<N, D: DimName, S> TranslationBase<N, D, S>
    where N: Scalar,
          S: Storage<N, D, U1> {
    /// Creates a new translation from the given vector.
    #[inline]
    pub fn from_vector(vector: ColumnVector<N, D, S>) -> TranslationBase<N, D, S> {
        TranslationBase {
            vector: vector
        }
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse(&self) -> OwnedTranslation<N, D, S>
        where N: ClosedNeg {
        TranslationBase::from_vector(-&self.vector)
    }

    /// Converts this translation into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedSquareMatrix<N, DimNameSum<D, U1>, S::Alloc>
        where N: Zero + One,
              D: DimNameAdd<U1>,
              S::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
        let mut res = OwnedSquareMatrix::<N, _, S::Alloc>::identity();
        res.fixed_slice_mut::<D, U1>(0, D::dim()).copy_from(&self.vector);

        res
    }
}


impl<N, D: DimName, S> TranslationBase<N, D, S>
    where N: Scalar + ClosedNeg,
          S: StorageMut<N, D, U1> {
    /// Inverts `self` in-place.
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.vector.neg_mut()
    }
}

impl<N, D: DimName, S> Eq for TranslationBase<N, D, S>
    where N: Scalar + Eq,
          S: Storage<N, D, U1> {
}

impl<N, D: DimName, S> PartialEq for TranslationBase<N, D, S>
    where N: Scalar + PartialEq,
          S: Storage<N, D, U1> {
    #[inline]
    fn eq(&self, right: &TranslationBase<N, D, S>) -> bool {
        self.vector == right.vector
    }
}

impl<N, D: DimName, S> ApproxEq for TranslationBase<N, D, S>
    where N: Scalar + ApproxEq,
          S: Storage<N, D, U1>,
          N::Epsilon: Copy {
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.vector.relative_eq(&other.vector, epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.vector.ulps_eq(&other.vector, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N, D: DimName, S> fmt::Display for TranslationBase<N, D, S>
    where N: Real + fmt::Display,
          S: Storage<N, D, U1>,
          S::Alloc: Allocator<usize, D, U1> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        try!(writeln!(f, "TranslationBase {{"));
        try!(write!(f, "{:.*}", precision, self.vector));
        writeln!(f, "}}")
    }
}


//          //         /*
//          //          *
//          //          * Absolute
//          //          *
//          //          */
//          //         impl<N: Absolute> Absolute for $t<N> {
//          //             type AbsoluteValue = $submatrix<N::AbsoluteValue>;
//          //
//          //             #[inline]
//          //             fn abs(m: &$t<N>) -> $submatrix<N::AbsoluteValue> {
//          //                 Absolute::abs(&m.submatrix)
//          //             }
//          //         }
//          */
