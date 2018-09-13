use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::{One, Zero};
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::{ClosedNeg, Real};

use base::allocator::Allocator;
use base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use base::storage::Owned;
use base::{DefaultAllocator, MatrixN, Scalar, VectorN};

/// A translation.
#[repr(C)]
#[derive(Debug)]
pub struct Translation<N: Scalar, D: DimName>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// The translation coordinates, i.e., how much is added to a point's coordinates when it is
    /// translated.
    pub vector: VectorN<N, D>,
}

impl<N: Scalar + hash::Hash, D: DimName + hash::Hash> hash::Hash for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.vector.hash(state)
    }
}

impl<N: Scalar, D: DimName> Copy for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Copy,
{
}

impl<N: Scalar, D: DimName> Clone for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Translation::from_vector(self.vector.clone())
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N, D> Abomonation for Translation<N, D>
where
    N: Scalar,
    D: DimName,
    VectorN<N, D>: Abomonation,
    DefaultAllocator: Allocator<N, D>,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.vector.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.vector.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.vector.exhume(bytes)
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: Scalar, D: DimName> Serialize for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.vector.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: Scalar, D: DimName> Deserialize<'a> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = VectorN::<N, D>::deserialize(deserializer)?;

        Ok(Translation::from_vector(matrix))
    }
}

impl<N: Scalar, D: DimName> Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new translation from the given vector.
    #[inline]
    pub fn from_vector(vector: VectorN<N, D>) -> Translation<N, D> {
        Translation { vector: vector }
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse(&self) -> Translation<N, D>
    where
        N: ClosedNeg,
    {
        Translation::from_vector(-&self.vector)
    }

    /// Converts this translation into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, DimNameSum<D, U1>>
    where
        N: Zero + One,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    {
        let mut res = MatrixN::<N, DimNameSum<D, U1>>::identity();
        res.fixed_slice_mut::<D, U1>(0, D::dim())
            .copy_from(&self.vector);

        res
    }

    /// Inverts `self` in-place.
    #[inline]
    pub fn inverse_mut(&mut self)
    where
        N: ClosedNeg,
    {
        self.vector.neg_mut()
    }
}

impl<N: Scalar + Eq, D: DimName> Eq for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
}

impl<N: Scalar + PartialEq, D: DimName> PartialEq for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn eq(&self, right: &Translation<N, D>) -> bool {
        self.vector == right.vector
    }
}

impl<N: Scalar + AbsDiffEq, D: DimName> AbsDiffEq for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.vector.abs_diff_eq(&other.vector, epsilon)
    }
}

impl<N: Scalar + RelativeEq, D: DimName> RelativeEq for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
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
        self.vector
            .relative_eq(&other.vector, epsilon, max_relative)
    }
}

impl<N: Scalar + UlpsEq, D: DimName> UlpsEq for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
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
impl<N: Real + fmt::Display, D: DimName> fmt::Display for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D> + Allocator<usize, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        try!(writeln!(f, "Translation {{"));
        try!(write!(f, "{:.*}", precision, self.vector));
        writeln!(f, "}}")
    }
}
