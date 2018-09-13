use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::{One, Zero};
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};

#[cfg(feature = "serde-serialize")]
use base::storage::Owned;

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::Real;

use base::allocator::Allocator;
use base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use base::{DefaultAllocator, MatrixN, Scalar};

/// A rotation matrix.
#[repr(C)]
#[derive(Debug)]
pub struct Rotation<N: Scalar, D: DimName>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    matrix: MatrixN<N, D>,
}

impl<N: Scalar + hash::Hash, D: DimName + hash::Hash> hash::Hash for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    <DefaultAllocator as Allocator<N, D, D>>::Buffer: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.matrix.hash(state)
    }
}

impl<N: Scalar, D: DimName> Copy for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    <DefaultAllocator as Allocator<N, D, D>>::Buffer: Copy,
{
}

impl<N: Scalar, D: DimName> Clone for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    <DefaultAllocator as Allocator<N, D, D>>::Buffer: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Rotation::from_matrix_unchecked(self.matrix.clone())
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N, D> Abomonation for Rotation<N, D>
where
    N: Scalar,
    D: DimName,
    MatrixN<N, D>: Abomonation,
    DefaultAllocator: Allocator<N, D, D>,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.matrix.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.matrix.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.matrix.exhume(bytes)
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: Scalar, D: DimName> Serialize for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    Owned<N, D, D>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: Scalar, D: DimName> Deserialize<'a> for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    Owned<N, D, D>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = MatrixN::<N, D>::deserialize(deserializer)?;

        Ok(Rotation::from_matrix_unchecked(matrix))
    }
}

impl<N: Scalar, D: DimName> Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// A reference to the underlying matrix representation of this rotation.
    #[inline]
    pub fn matrix(&self) -> &MatrixN<N, D> {
        &self.matrix
    }

    /// A mutable reference to the underlying matrix representation of this rotation.
    ///
    /// This is unsafe because this allows the user to replace the matrix by another one that is
    /// non-square, non-inversible, or non-orthonormal. If one of those properties is broken,
    /// subsequent method calls may be UB.
    #[inline]
    pub unsafe fn matrix_mut(&mut self) -> &mut MatrixN<N, D> {
        &mut self.matrix
    }

    /// Unwraps the underlying matrix.
    #[inline]
    pub fn unwrap(self) -> MatrixN<N, D> {
        self.matrix
    }

    /// Converts this rotation into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, DimNameSum<D, U1>>
    where
        N: Zero + One,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    {
        let mut res = MatrixN::<N, DimNameSum<D, U1>>::identity();
        res.fixed_slice_mut::<D, D>(0, 0).copy_from(&self.matrix);

        res
    }

    /// Creates a new rotation from the given square matrix.
    ///
    /// The matrix squareness is checked but not its orthonormality.
    #[inline]
    pub fn from_matrix_unchecked(matrix: MatrixN<N, D>) -> Rotation<N, D> {
        assert!(
            matrix.is_square(),
            "Unable to create a rotation from a non-square matrix."
        );

        Rotation { matrix: matrix }
    }

    /// Transposes `self`.
    #[inline]
    pub fn transpose(&self) -> Rotation<N, D> {
        Rotation::from_matrix_unchecked(self.matrix.transpose())
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse(&self) -> Rotation<N, D> {
        self.transpose()
    }

    /// Transposes `self` in-place.
    #[inline]
    pub fn transpose_mut(&mut self) {
        self.matrix.transpose_mut()
    }

    /// Inverts `self` in-place.
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.transpose_mut()
    }
}

impl<N: Scalar + Eq, D: DimName> Eq for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
{
}

impl<N: Scalar + PartialEq, D: DimName> PartialEq for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    #[inline]
    fn eq(&self, right: &Rotation<N, D>) -> bool {
        self.matrix == right.matrix
    }
}

impl<N, D: DimName> AbsDiffEq for Rotation<N, D>
where
    N: Scalar + AbsDiffEq,
    DefaultAllocator: Allocator<N, D, D>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}

impl<N, D: DimName> RelativeEq for Rotation<N, D>
where
    N: Scalar + RelativeEq,
    DefaultAllocator: Allocator<N, D, D>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}

impl<N, D: DimName> UlpsEq for Rotation<N, D>
where
    N: Scalar + UlpsEq,
    DefaultAllocator: Allocator<N, D, D>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.matrix.ulps_eq(&other.matrix, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N, D: DimName> fmt::Display for Rotation<N, D>
where
    N: Real + fmt::Display,
    DefaultAllocator: Allocator<N, D, D> + Allocator<usize, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        try!(writeln!(f, "Rotation matrix {{"));
        try!(write!(f, "{:.*}", precision, self.matrix));
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
