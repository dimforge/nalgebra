#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::One;
#[cfg(feature = "rand-no-std")]
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use simba::scalar::{ClosedMulAssign, SupersetOf};

use crate::base::{SVector, Scalar};
use crate::geometry::Scale;

impl<T: Scalar, const D: usize> Scale<T, D> {
    /// Creates a new identity scale transformation.
    ///
    /// An identity scale has all factors set to 1, meaning it doesn't change anything when applied.
    /// This is useful as a starting point for transformations or as a neutral element in
    /// transformation chains.
    ///
    /// # What is an Identity Scale?
    ///
    /// An identity scale is the scaling equivalent of multiplying by 1:
    /// - In 2D: `Scale2::identity()` = `Scale2::new(1.0, 1.0)`
    /// - In 3D: `Scale3::identity()` = `Scale3::new(1.0, 1.0, 1.0)`
    ///
    /// Points transformed by an identity scale remain unchanged.
    ///
    /// # Examples
    ///
    /// ## Basic usage
    /// ```
    /// # use nalgebra::{Scale3, Point3};
    /// let identity = Scale3::identity();
    /// let point = Point3::new(5.0, 10.0, 15.0);
    ///
    /// // Point remains unchanged
    /// assert_eq!(identity.transform_point(&point), point);
    /// ```
    ///
    /// ## Composing with identity
    /// ```
    /// # use nalgebra::Scale2;
    /// let scale = Scale2::new(2.0, 3.0);
    /// let identity = Scale2::identity();
    ///
    /// // Composing with identity doesn't change the scale
    /// assert_eq!(scale * identity, scale);
    /// assert_eq!(identity * scale, scale);
    /// ```
    ///
    /// ## Using as a starting point
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // Start with identity, then modify based on conditions
    /// let mut dynamic_scale = Scale2::identity();
    ///
    /// let should_double_width = true;
    /// if should_double_width {
    ///     dynamic_scale.vector.x = 2.0;
    /// }
    ///
    /// let point = Point2::new(10.0, 20.0);
    /// let result = dynamic_scale.transform_point(&point);
    /// assert_eq!(result, Point2::new(20.0, 20.0));  // Only X doubled
    /// ```
    ///
    /// ## Inverse property
    /// ```
    /// # use nalgebra::Scale3;
    /// let scale = Scale3::new(2.0, 4.0, 8.0);
    /// let inverse = scale.try_inverse().unwrap();
    ///
    /// // Scale composed with its inverse gives identity
    /// assert_eq!(scale * inverse, Scale3::identity());
    /// assert_eq!(inverse * scale, Scale3::identity());
    /// ```
    ///
    /// ## Works in all dimensions
    /// ```
    /// # use nalgebra::{Point2, Scale2};
    /// let identity_2d = Scale2::identity();
    /// let p2 = Point2::new(1.0, 2.0);
    /// assert_eq!(identity_2d * p2, p2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new`](Self::new) - Create a scale with specific factors
    /// - [`from`](From) - Create from a vector or array
    #[inline]
    pub fn identity() -> Scale<T, D>
    where
        T: One,
    {
        Scale::from(SVector::from_element(T::one()))
    }

    /// Casts the components of this scale to another type.
    ///
    /// This converts all scale factors from type `T` to type `To`. This is useful when you need
    /// to change precision (e.g., `f64` to `f32`) or work with different numeric types.
    ///
    /// # Examples
    ///
    /// ## Convert from f64 to f32
    /// ```
    /// # use nalgebra::Scale2;
    /// let scale_f64 = Scale2::new(1.5f64, 2.5f64);
    /// let scale_f32 = scale_f64.cast::<f32>();
    ///
    /// assert_eq!(scale_f32, Scale2::new(1.5f32, 2.5f32));
    /// ```
    ///
    /// ## Convert from f32 to f64 for higher precision
    /// ```
    /// # use nalgebra::Scale3;
    /// let scale_f32 = Scale3::new(1.0f32, 2.0f32, 3.0f32);
    /// let scale_f64 = scale_f32.cast::<f64>();
    ///
    /// assert_eq!(scale_f64, Scale3::new(1.0f64, 2.0f64, 3.0f64));
    /// ```
    ///
    /// ## Practical use: mixing precisions
    /// ```
    /// # use nalgebra::{Scale2, Point2};
    /// // High-precision scale factors
    /// let precise_scale = Scale2::new(1.333333333333f64, 0.666666666666f64);
    ///
    /// // Convert to f32 for rendering
    /// let render_scale = precise_scale.cast::<f32>();
    ///
    /// // Use with f32 points
    /// let point = Point2::new(3.0f32, 6.0f32);
    /// let scaled = render_scale.transform_point(&point);
    /// ```
    ///
    /// # Type Requirements
    ///
    /// The target type `To` must be a superset of the source type `T`, meaning `To` can represent
    /// all values that `T` can represent. This is automatically satisfied for standard numeric
    /// conversions like:
    /// - Integer to floating-point (i32 to f64)
    /// - Lower to higher precision (f32 to f64)
    ///
    /// # See Also
    ///
    /// - [`new`](Self::new) - Create a scale with specific type from the start
    pub fn cast<To: Scalar>(self) -> Scale<To, D>
    where
        Scale<To, D>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T: Scalar + One + ClosedMulAssign, const D: usize> One for Scale<T, D> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: Scalar, const D: usize> Distribution<Scale<T, D>> for StandardUniform
where
    StandardUniform: Distribution<T>,
{
    /// Generate an arbitrary random variate for testing purposes.
    #[inline]
    fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> Scale<T, D> {
        Scale::from(rng.random::<SVector<T, D>>())
    }
}

#[cfg(feature = "arbitrary")]
impl<T: Scalar + Arbitrary + Send, const D: usize> Arbitrary for Scale<T, D>
where
    Owned<T, crate::Const<D>>: Send,
{
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        let v: SVector<T, D> = Arbitrary::arbitrary(rng);
        Self::from(v)
    }
}

/*
 *
 * Small Scale construction from components.
 *
 */
macro_rules! componentwise_constructors_impl(
    ($($doc: expr_2021; $D: expr_2021, $($args: ident:$irow: expr_2021),*);* $(;)*) => {$(
        impl<T> Scale<T, $D>
             {
            #[doc = "Creates a new scale transformation with the given factors for each axis."]
            #[doc = ""]
            #[doc = "Each parameter represents the scale factor for the corresponding axis. A factor of 1.0 leaves"]
            #[doc = "that axis unchanged, values > 1.0 make objects larger, and values < 1.0 make them smaller."]
            #[doc = "Negative values flip the object along that axis."]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "## Basic usage"]
            #[doc = "```"]
            #[doc = $doc]
            #[doc = "```"]
            #[doc = ""]
            #[doc = "## Common use cases"]
            #[doc = "```"]
            #[doc = "# use nalgebra::{Scale2, Point2};"]
            #[doc = "// Double the size uniformly"]
            #[doc = "let uniform = Scale2::new(2.0, 2.0);"]
            #[doc = ""]
            #[doc = "// Stretch horizontally, compress vertically (non-uniform)"]
            #[doc = "let non_uniform = Scale2::new(2.0, 0.5);"]
            #[doc = ""]
            #[doc = "// Flip horizontally using negative scale"]
            #[doc = "let flip_h = Scale2::new(-1.0, 1.0);"]
            #[doc = "let point = Point2::new(5.0, 10.0);"]
            #[doc = "assert_eq!(flip_h.transform_point(&point), Point2::new(-5.0, 10.0));"]
            #[doc = "```"]
            #[doc = ""]
            #[doc = "# See Also"]
            #[doc = ""]
            #[doc = "- [`identity`](Self::identity) - Creates a scale with all factors = 1"]
            #[doc = "- [`from`](From) - Create from a vector or array"]
            #[inline]
            pub const fn new($($args: T),*) -> Self {
                Self { vector: SVector::<T, $D>::new($($args),*) }
            }
        }
    )*}
);

componentwise_constructors_impl!(
    "# use nalgebra::Scale1;\nlet t = Scale1::new(1.0);\nassert!(t.vector.x == 1.0);";
    1, x:0;
    "# use nalgebra::Scale2;\nlet t = Scale2::new(1.0, 2.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0);";
    2, x:0, y:1;
    "# use nalgebra::Scale3;\nlet t = Scale3::new(1.0, 2.0, 3.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0 && t.vector.z == 3.0);";
    3, x:0, y:1, z:2;
    "# use nalgebra::Scale4;\nlet t = Scale4::new(1.0, 2.0, 3.0, 4.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0 && t.vector.z == 3.0 && t.vector.w == 4.0);";
    4, x:0, y:1, z:2, w:3;
    "# use nalgebra::Scale5;\nlet t = Scale5::new(1.0, 2.0, 3.0, 4.0, 5.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0 && t.vector.z == 3.0 && t.vector.w == 4.0 && t.vector.a == 5.0);";
    5, x:0, y:1, z:2, w:3, a:4;
    "# use nalgebra::Scale6;\nlet t = Scale6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0 && t.vector.z == 3.0 && t.vector.w == 4.0 && t.vector.a == 5.0 && t.vector.b == 6.0);";
    6, x:0, y:1, z:2, w:3, a:4, b:5;
);
