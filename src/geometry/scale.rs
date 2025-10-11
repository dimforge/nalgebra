// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::{One, Zero};
use std::fmt;
use std::hash;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ClosedDivAssign;
use crate::ClosedMulAssign;
use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::storage::Owned;
use crate::base::{Const, DefaultAllocator, OMatrix, OVector, SVector, Scalar};

use crate::geometry::Point;

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

/// A scale transformation that supports non-uniform scaling.
///
/// # What is Non-Uniform Scaling?
///
/// Non-uniform scaling allows different scale factors along each axis. This is more flexible
/// than uniform scaling (where all axes scale by the same amount).
///
/// For example, in 2D:
/// - Uniform scaling by 2: `Scale2::new(2.0, 2.0)` makes objects twice as large in all directions
/// - Non-uniform scaling: `Scale2::new(2.0, 0.5)` stretches horizontally and squashes vertically
///
/// # Common Use Cases
///
/// - **Sprite scaling**: Different width/height scaling for game sprites
/// - **Hitbox adjustment**: Resize collision boxes independently per axis
/// - **Ellipse creation**: Transform circles into ellipses by scaling axes differently
/// - **Squash and stretch**: Classic animation technique for organic motion
///
/// # Example
/// ```
/// # use nalgebra::{Scale2, Point2};
/// // Create a non-uniform scale (2x wider, 0.5x shorter)
/// let scale = Scale2::new(2.0, 0.5);
///
/// // Apply to a point
/// let point = Point2::new(1.0, 4.0);
/// let scaled = scale.transform_point(&point);
/// assert_eq!(scaled, Point2::new(2.0, 2.0));
/// ```
#[repr(C)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Scale<T::Archived, D>",
        bound(archive = "
        T: rkyv::Archive,
        SVector<T, D>: rkyv::Archive<Archived = SVector<T::Archived, D>>
    ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Copy, Clone)]
pub struct Scale<T, const D: usize> {
    /// The scale coordinates, i.e., how much is multiplied to a point's coordinates when it is
    /// scaled.
    pub vector: SVector<T, D>,
}

impl<T: fmt::Debug, const D: usize> fmt::Debug for Scale<T, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.vector.as_slice().fmt(formatter)
    }
}

impl<T: Scalar + hash::Hash, const D: usize> hash::Hash for Scale<T, D>
where
    Owned<T, Const<D>>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.vector.hash(state)
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const D: usize> bytemuck::Zeroable for Scale<T, D>
where
    T: Scalar + bytemuck::Zeroable,
    SVector<T, D>: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const D: usize> bytemuck::Pod for Scale<T, D>
where
    T: Scalar + bytemuck::Pod,
    SVector<T, D>: bytemuck::Pod,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: Scalar, const D: usize> Serialize for Scale<T, D>
where
    Owned<T, Const<D>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.vector.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: Scalar, const D: usize> Deserialize<'a> for Scale<T, D>
where
    Owned<T, Const<D>>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = SVector::<T, D>::deserialize(deserializer)?;

        Ok(Scale::from(matrix))
    }
}

impl<T: Scalar, const D: usize> Scale<T, D> {
    /// Attempts to compute the inverse of this scale transformation.
    ///
    /// The inverse scale reverses the scaling operation. For a scale with factors `(sx, sy, sz)`,
    /// the inverse has factors `(1/sx, 1/sy, 1/sz)`. This is useful for "undoing" a scaling
    /// transformation or for transforming points in the opposite direction.
    ///
    /// Returns `None` if any scale factor is zero (since division by zero is undefined).
    ///
    /// # Examples
    ///
    /// ## Basic usage in 3D
    /// ```
    /// # use nalgebra::{Scale3, Point3};
    /// let scale = Scale3::new(2.0, 4.0, 8.0);
    /// let inverse = scale.try_inverse().unwrap();
    ///
    /// // The inverse reverses the scaling
    /// assert_eq!(inverse, Scale3::new(0.5, 0.25, 0.125));
    ///
    /// // Applying scale then inverse returns identity
    /// assert_eq!(scale * inverse, Scale3::identity());
    /// assert_eq!(inverse * scale, Scale3::identity());
    /// ```
    ///
    /// ## 2D sprite scaling example
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // Scale a sprite to 3x width, 2x height
    /// let scale = Scale2::new(3.0, 2.0);
    /// let sprite_corner = Point2::new(1.0, 1.0);
    /// let scaled_corner = scale.transform_point(&sprite_corner);
    ///
    /// // Undo the scaling with the inverse
    /// let inverse = scale.try_inverse().unwrap();
    /// let original = inverse.transform_point(&scaled_corner);
    /// assert_eq!(original, sprite_corner);
    /// ```
    ///
    /// ## Handling zero scale factors
    /// ```
    /// # use nalgebra::Scale2;
    /// // Cannot invert a scale with zero factor
    /// let scale = Scale2::new(0.0, 2.0);
    /// assert_eq!(scale.try_inverse(), None);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`inverse_unchecked`](Self::inverse_unchecked) - Faster version without zero-check (unsafe)
    /// - [`pseudo_inverse`](Self::pseudo_inverse) - Returns zero for zero scale factors
    /// - [`try_inverse_mut`](Self::try_inverse_mut) - In-place version
    #[inline]
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn try_inverse(&self) -> Option<Scale<T, D>>
    where
        T: ClosedDivAssign + One + Zero,
    {
        for i in 0..D {
            if self.vector[i] == T::zero() {
                return None;
            }
        }
        Some(self.vector.map(|e| T::one() / e).into())
    }

    /// Computes the inverse of this scale transformation without checking for zero scale factors.
    ///
    /// This is a faster alternative to [`try_inverse`](Self::try_inverse) that skips the
    /// zero-check. The inverse scale has factors `(1/sx, 1/sy, 1/sz)` for a scale with
    /// factors `(sx, sy, sz)`.
    ///
    /// # Safety
    ///
    /// This function performs division by the scale factors without checking if they are zero.
    /// Calling this with any zero scale factor will result in undefined behavior (infinity or NaN
    /// for floating-point types). Only use this function if you can guarantee that all scale
    /// factors are non-zero.
    ///
    /// # Examples
    ///
    /// ## Basic usage (safe scale factors)
    /// ```
    /// # use nalgebra::Scale3;
    /// let scale = Scale3::new(2.0, 4.0, 8.0);
    ///
    /// unsafe {
    ///     let inverse = scale.inverse_unchecked();
    ///     assert_eq!(inverse, Scale3::new(0.5, 0.25, 0.125));
    ///
    ///     // Composing with inverse gives identity
    ///     assert_eq!(scale * inverse, Scale3::identity());
    ///     assert_eq!(inverse * scale, Scale3::identity());
    /// }
    /// ```
    ///
    /// ## Performance-critical code
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // In a hot loop where you know scale factors are never zero
    /// let scale = Scale2::new(1.5, 2.5);
    /// let points = vec![Point2::new(10.0, 20.0), Point2::new(30.0, 40.0)];
    ///
    /// unsafe {
    ///     let inverse = scale.inverse_unchecked();
    ///     for point in &points {
    ///         let transformed = inverse.transform_point(point);
    ///         // Process transformed point...
    ///     }
    /// }
    /// ```
    ///
    /// ## Undefined behavior with zero (DO NOT DO THIS)
    /// ```no_run
    /// # use nalgebra::Scale2;
    /// let scale = Scale2::new(0.0, 2.0);  // Contains zero!
    ///
    /// unsafe {
    ///     // This is undefined behavior - will produce infinity or NaN
    ///     let inverse = scale.inverse_unchecked();
    ///     // inverse.vector.x will be infinity or NaN
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_inverse`](Self::try_inverse) - Safe version that checks for zeros
    /// - [`pseudo_inverse`](Self::pseudo_inverse) - Handles zeros by leaving them as zero
    #[inline]
    #[must_use]
    pub unsafe fn inverse_unchecked(&self) -> Scale<T, D>
    where
        T: ClosedDivAssign + One,
    {
        self.vector.map(|e| T::one() / e).into()
    }

    /// Computes the pseudo-inverse of this scale transformation.
    ///
    /// The pseudo-inverse is similar to the regular inverse, but handles zero scale factors
    /// gracefully. For non-zero factors, it computes `1/factor`. For zero factors, it leaves
    /// them as zero instead of failing or producing infinity/NaN.
    ///
    /// This is useful when you want to invert a scale but some axes might have zero scaling
    /// (effectively collapsing to a lower dimension).
    ///
    /// # How It Works
    ///
    /// For a scale with factors `(sx, sy, sz)`:
    /// - Non-zero factors are inverted: `sx -> 1/sx`
    /// - Zero factors remain zero: `0 -> 0`
    ///
    /// # Examples
    ///
    /// ## Basic usage with non-zero factors
    /// ```
    /// # use nalgebra::Scale3;
    /// let scale = Scale3::new(2.0, 4.0, 8.0);
    /// let pseudo_inv = scale.pseudo_inverse();
    ///
    /// // Acts like regular inverse for non-zero factors
    /// assert_eq!(pseudo_inv, Scale3::new(0.5, 0.25, 0.125));
    /// assert_eq!(scale * pseudo_inv, Scale3::identity());
    /// ```
    ///
    /// ## Handling zero scale factors
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // Scale that collapses the x-axis to zero
    /// let scale = Scale2::new(0.0, 2.0);
    /// let pseudo_inv = scale.pseudo_inverse();
    ///
    /// // Zero factors remain zero, non-zero factors are inverted
    /// assert_eq!(pseudo_inv, Scale2::new(0.0, 0.5));
    ///
    /// // Composing doesn't give full identity (x-axis stays collapsed)
    /// assert_eq!(scale * pseudo_inv, Scale2::new(0.0, 1.0));
    /// ```
    ///
    /// ## Projection use case
    /// ```
    /// # use nalgebra::{Scale3, Point3};
    /// // Flatten geometry onto XY plane (zero Z scaling)
    /// let flatten = Scale3::new(1.0, 1.0, 0.0);
    /// let point = Point3::new(5.0, 10.0, 15.0);
    /// let flattened = flatten.transform_point(&point);
    ///
    /// assert_eq!(flattened, Point3::new(5.0, 10.0, 0.0));
    ///
    /// // Pseudo-inverse keeps XY, leaves Z at zero
    /// let pseudo_inv = flatten.pseudo_inverse();
    /// assert_eq!(pseudo_inv, Scale3::new(1.0, 1.0, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_inverse`](Self::try_inverse) - Returns `None` if any factor is zero
    /// - [`inverse_unchecked`](Self::inverse_unchecked) - Unsafe version without checks
    #[inline]
    #[must_use]
    pub fn pseudo_inverse(&self) -> Scale<T, D>
    where
        T: ClosedDivAssign + One + Zero,
    {
        self.vector
            .map(|e| {
                if e != T::zero() {
                    T::one() / e
                } else {
                    T::zero()
                }
            })
            .into()
    }

    /// Converts this scale transformation into its equivalent homogeneous transformation matrix.
    ///
    /// A homogeneous matrix is a standard way to represent transformations in computer graphics.
    /// For a scale, this produces a diagonal matrix with the scale factors on the diagonal and
    /// 1 in the bottom-right corner.
    ///
    /// # Why Use Homogeneous Matrices?
    ///
    /// Homogeneous matrices allow you to:
    /// - Combine multiple transformations (scale, rotation, translation) by matrix multiplication
    /// - Use with graphics APIs that expect matrix form
    /// - Apply transformations using standard matrix-vector multiplication
    ///
    /// # Matrix Structure
    ///
    /// For a 3D scale with factors `(sx, sy, sz)`, the matrix is:
    /// ```text
    /// [sx  0   0   0]
    /// [0   sy  0   0]
    /// [0   0   sz  0]
    /// [0   0   0   1]
    /// ```
    ///
    /// # Examples
    ///
    /// ## 3D scale to matrix
    /// ```
    /// # use nalgebra::{Scale3, Matrix4};
    /// let scale = Scale3::new(2.0, 3.0, 4.0);
    /// let matrix = scale.to_homogeneous();
    ///
    /// let expected = Matrix4::new(
    ///     2.0, 0.0, 0.0, 0.0,
    ///     0.0, 3.0, 0.0, 0.0,
    ///     0.0, 0.0, 4.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0
    /// );
    /// assert_eq!(matrix, expected);
    /// ```
    ///
    /// ## 2D scale to matrix
    /// ```
    /// # use nalgebra::{Scale2, Matrix3};
    /// let scale = Scale2::new(1.5, 0.5);
    /// let matrix = scale.to_homogeneous();
    ///
    /// let expected = Matrix3::new(
    ///     1.5, 0.0, 0.0,
    ///     0.0, 0.5, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    /// assert_eq!(matrix, expected);
    /// ```
    ///
    /// ## Combining with other transformations
    /// ```
    /// # use nalgebra::{Scale3, Translation3, Matrix4};
    /// let scale = Scale3::new(2.0, 2.0, 2.0);
    /// let translation = Translation3::new(10.0, 20.0, 30.0);
    ///
    /// // Combine transformations by multiplying their matrices
    /// let combined = translation.to_homogeneous() * scale.to_homogeneous();
    ///
    /// // This scales first, then translates
    /// let expected = Matrix4::new(
    ///     2.0, 0.0, 0.0, 10.0,
    ///     0.0, 2.0, 0.0, 20.0,
    ///     0.0, 0.0, 2.0, 30.0,
    ///     0.0, 0.0, 0.0, 1.0
    /// );
    /// assert_eq!(combined, expected);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`From<Scale>`](From) - Can convert implicitly using `.into()`
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        T: Zero + One + Clone,
        Const<D>: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
            + Allocator<DimNameSum<Const<D>, U1>, U1>,
    {
        // TODO: use self.vector.push() instead. We can’t right now because
        //       that would require the DimAdd bound (but here we use DimNameAdd).
        //       This should be fixable once Rust gets a more complete support of
        //       const-generics.
        let mut v = OVector::from_element(T::one());
        for i in 0..D {
            v[i] = self.vector[i].clone();
        }
        OMatrix::from_diagonal(&v)
    }

    /// Attempts to invert this scale transformation in-place.
    ///
    /// This is the in-place version of [`try_inverse`](Self::try_inverse). Instead of returning
    /// a new scale, it modifies `self` to become its inverse. This is more efficient when you
    /// don't need to keep the original scale.
    ///
    /// Returns `true` if successful, `false` if any scale factor is zero (in which case `self`
    /// is left unchanged).
    ///
    /// # Examples
    ///
    /// ## Basic usage
    /// ```
    /// # use nalgebra::Scale3;
    /// let original = Scale3::new(2.0, 4.0, 8.0);
    /// let mut scale = original;
    ///
    /// assert!(scale.try_inverse_mut());
    /// assert_eq!(scale, Scale3::new(0.5, 0.25, 0.125));
    ///
    /// // Verify it's the inverse
    /// assert_eq!(original * scale, Scale3::identity());
    /// ```
    ///
    /// ## Animation: bounce effect
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // Animate a sprite bouncing (squash and stretch)
    /// let mut sprite_scale = Scale2::new(1.0, 1.5);  // Stretched up
    /// let sprite_pos = Point2::new(10.0, 20.0);
    ///
    /// // Scale the sprite
    /// let scaled = sprite_scale.transform_point(&sprite_pos);
    ///
    /// // Later: return to normal by inverting the scale
    /// sprite_scale.try_inverse_mut();
    /// let normal = sprite_scale.transform_point(&scaled);
    /// assert_eq!(normal, sprite_pos);
    /// ```
    ///
    /// ## Handling zero factors
    /// ```
    /// # use nalgebra::Scale2;
    /// let mut scale = Scale2::new(0.0, 2.0);
    ///
    /// // Cannot invert due to zero factor
    /// assert!(!scale.try_inverse_mut());
    ///
    /// // Scale remains unchanged
    /// assert_eq!(scale, Scale2::new(0.0, 2.0));
    /// ```
    ///
    /// ## Performance comparison
    /// ```
    /// # use nalgebra::Scale3;
    /// let mut scale1 = Scale3::new(2.0, 3.0, 4.0);
    /// let scale2 = Scale3::new(2.0, 3.0, 4.0);
    ///
    /// // In-place: modifies existing scale (more efficient)
    /// scale1.try_inverse_mut();
    ///
    /// // Creates new scale (allocates new memory)
    /// let scale2_inv = scale2.try_inverse().unwrap();
    ///
    /// assert_eq!(scale1, scale2_inv);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_inverse`](Self::try_inverse) - Returns a new inverted scale
    /// - [`inverse_unchecked`](Self::inverse_unchecked) - Unsafe version without zero-check
    /// - [`pseudo_inverse`](Self::pseudo_inverse) - Handles zeros by leaving them as zero
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool
    where
        T: ClosedDivAssign + One + Zero,
    {
        match self.try_inverse() {
            Some(v) => {
                self.vector = v.vector;
                true
            }
            None => false,
        }
    }
}

impl<T: Scalar + ClosedMulAssign, const D: usize> Scale<T, D> {
    /// Applies this scale transformation to a point.
    ///
    /// Each coordinate of the point is multiplied by the corresponding scale factor.
    /// For a scale `(sx, sy, sz)` and point `(x, y, z)`, the result is `(sx*x, sy*y, sz*z)`.
    ///
    /// This is equivalent to the multiplication `self * pt`, but may be more readable.
    ///
    /// # Examples
    ///
    /// ## Basic transformation
    /// ```
    /// # use nalgebra::{Scale3, Point3};
    /// let scale = Scale3::new(2.0, 3.0, 4.0);
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let result = scale.transform_point(&point);
    ///
    /// assert_eq!(result, Point3::new(2.0, 6.0, 12.0));
    /// ```
    ///
    /// ## Game sprite scaling
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // Make a sprite 2x wider and 1.5x taller
    /// let sprite_scale = Scale2::new(2.0, 1.5);
    ///
    /// // Transform sprite corners
    /// let top_left = Point2::new(0.0, 0.0);
    /// let bottom_right = Point2::new(32.0, 32.0);  // Original 32x32 sprite
    ///
    /// let scaled_tl = sprite_scale.transform_point(&top_left);
    /// let scaled_br = sprite_scale.transform_point(&bottom_right);
    ///
    /// assert_eq!(scaled_tl, Point2::new(0.0, 0.0));
    /// assert_eq!(scaled_br, Point2::new(64.0, 48.0));  // Now 64x48
    /// ```
    ///
    /// ## Creating an ellipse from a circle
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// use std::f64::consts::PI;
    ///
    /// // Start with a unit circle point
    /// let angle = PI / 4.0;  // 45 degrees
    /// let circle_point = Point2::new(angle.cos(), angle.sin());
    ///
    /// // Stretch to create an ellipse (3:1 ratio)
    /// let ellipse_scale = Scale2::new(3.0, 1.0);
    /// let ellipse_point = ellipse_scale.transform_point(&circle_point);
    ///
    /// // X coordinate is 3x larger, Y stays the same
    /// assert!((ellipse_point.x - 3.0 * angle.cos()).abs() < 1e-10);
    /// assert!((ellipse_point.y - angle.sin()).abs() < 1e-10);
    /// ```
    ///
    /// ## Hitbox scaling
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // Scale down a hitbox to 80% of sprite size
    /// let hitbox_scale = Scale2::new(0.8, 0.8);
    ///
    /// let sprite_corner = Point2::new(50.0, 50.0);
    /// let hitbox_corner = hitbox_scale.transform_point(&sprite_corner);
    ///
    /// assert_eq!(hitbox_corner, Point2::new(40.0, 40.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_inverse_transform_point`](Self::try_inverse_transform_point) - Transform by the inverse scale
    /// - Multiplication operator `*` - Alternative syntax: `scale * point`
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self * pt
    }
}

impl<T: Scalar + ClosedDivAssign + ClosedMulAssign + One + Zero, const D: usize> Scale<T, D> {
    /// Applies the inverse of this scale transformation to a point.
    ///
    /// This divides each coordinate of the point by the corresponding scale factor, effectively
    /// "undoing" the scale. For a scale `(sx, sy, sz)` and point `(x, y, z)`, the result is
    /// `(x/sx, y/sy, z/sz)`.
    ///
    /// Returns `None` if any scale factor is zero (since division by zero is undefined).
    ///
    /// # When to Use This
    ///
    /// - Converting from scaled space back to original space
    /// - Undoing a previous scaling operation
    /// - Working with inverse transformations without explicitly computing the inverse scale
    ///
    /// # Examples
    ///
    /// ## Basic inverse transformation
    /// ```
    /// # use nalgebra::{Scale3, Point3};
    /// let scale = Scale3::new(2.0, 4.0, 8.0);
    /// let scaled_point = Point3::new(10.0, 20.0, 40.0);
    ///
    /// // Transform by inverse scale
    /// let original = scale.try_inverse_transform_point(&scaled_point).unwrap();
    /// assert_eq!(original, Point3::new(5.0, 5.0, 5.0));
    /// ```
    ///
    /// ## Round-trip transformation
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// let scale = Scale2::new(3.0, 2.0);
    /// let original = Point2::new(5.0, 10.0);
    ///
    /// // Scale and then inverse-scale returns original
    /// let scaled = scale.transform_point(&original);
    /// let back = scale.try_inverse_transform_point(&scaled).unwrap();
    ///
    /// assert_eq!(back, original);
    /// ```
    ///
    /// ## Converting screen space to world space
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // Viewport is scaled 2x for hi-DPI display
    /// let viewport_scale = Scale2::new(2.0, 2.0);
    ///
    /// // Mouse click at screen position
    /// let screen_pos = Point2::new(400.0, 300.0);
    ///
    /// // Convert to world position
    /// let world_pos = viewport_scale.try_inverse_transform_point(&screen_pos).unwrap();
    /// assert_eq!(world_pos, Point2::new(200.0, 150.0));
    /// ```
    ///
    /// ## Handling zero scale factors
    /// ```
    /// # use nalgebra::{Scale3, Point3};
    /// let scale = Scale3::new(2.0, 0.0, 4.0);  // Zero Y scale
    /// let point = Point3::new(10.0, 20.0, 40.0);
    ///
    /// // Cannot inverse-transform due to zero factor
    /// assert_eq!(scale.try_inverse_transform_point(&point), None);
    /// ```
    ///
    /// ## Alternative: explicit inverse
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// let scale = Scale2::new(3.0, 4.0);
    /// let point = Point2::new(12.0, 16.0);
    ///
    /// // These two approaches are equivalent:
    /// let result1 = scale.try_inverse_transform_point(&point).unwrap();
    /// let result2 = scale.try_inverse().unwrap().transform_point(&point);
    ///
    /// assert_eq!(result1, result2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`transform_point`](Self::transform_point) - Apply the scale (not its inverse)
    /// - [`try_inverse`](Self::try_inverse) - Get the inverse scale explicitly
    #[inline]
    #[must_use]
    pub fn try_inverse_transform_point(&self, pt: &Point<T, D>) -> Option<Point<T, D>> {
        self.try_inverse().map(|s| s * pt)
    }
}

impl<T: Scalar + Eq, const D: usize> Eq for Scale<T, D> {}

impl<T: Scalar + PartialEq, const D: usize> PartialEq for Scale<T, D> {
    #[inline]
    fn eq(&self, right: &Scale<T, D>) -> bool {
        self.vector == right.vector
    }
}

impl<T: Scalar + AbsDiffEq, const D: usize> AbsDiffEq for Scale<T, D>
where
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.vector.abs_diff_eq(&other.vector, epsilon)
    }
}

impl<T: Scalar + RelativeEq, const D: usize> RelativeEq for Scale<T, D>
where
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
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

impl<T: Scalar + UlpsEq, const D: usize> UlpsEq for Scale<T, D>
where
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
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
impl<T: Scalar + fmt::Display, const D: usize> fmt::Display for Scale<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Scale {{")?;
        write!(f, "{:.*}", precision, self.vector)?;
        writeln!(f, "}}")
    }
}
