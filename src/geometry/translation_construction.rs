#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::{One, Zero};
#[cfg(feature = "rand-no-std")]
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use simba::scalar::{ClosedAddAssign, SupersetOf};

use crate::base::{SVector, Scalar};
use crate::geometry::Translation;

impl<T: Scalar + Zero, const D: usize> Default for Translation<T, D> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Scalar, const D: usize> Translation<T, D> {
    /// Creates a new identity translation that doesn't move points.
    ///
    /// The identity translation is the neutral element for translation composition - it leaves
    /// all points unchanged when applied. It's equivalent to a translation with zero offset
    /// in all dimensions.
    ///
    /// This is useful as a starting point or default value when you need a translation that
    /// does nothing, similar to how multiplying by 1 or adding 0 doesn't change a number.
    ///
    /// # Returns
    ///
    /// A translation with zero displacement in all dimensions.
    ///
    /// # Examples
    ///
    /// Basic usage in 2D:
    /// ```
    /// # use nalgebra::{Translation2, Point2};
    /// let t = Translation2::identity();
    /// let point = Point2::new(1.0, 2.0);
    ///
    /// // Identity translation doesn't move the point
    /// assert_eq!(t * point, point);
    /// ```
    ///
    /// Works in 3D and other dimensions:
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let t = Translation3::identity();
    /// let p = Point3::new(1.0, 2.0, 3.0);
    /// assert_eq!(t * p, p);
    /// ```
    ///
    /// Identity is neutral for composition:
    /// ```
    /// # use nalgebra::Translation2;
    /// let t = Translation2::new(5.0, 10.0);
    /// let identity = Translation2::identity();
    ///
    /// // Composing with identity doesn't change the translation
    /// assert_eq!(t * identity, t);
    /// assert_eq!(identity * t, t);
    /// ```
    ///
    /// Using as a default value:
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// // Initialize camera with no offset
    /// let mut camera_offset = Translation3::identity();
    ///
    /// let point = Point3::new(10.0, 20.0, 30.0);
    /// assert_eq!(camera_offset * point, point);
    ///
    /// // Later, update the offset
    /// camera_offset = Translation3::new(5.0, 0.0, 0.0);
    /// assert_eq!(camera_offset * point, Point3::new(15.0, 20.0, 30.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Translation::new`] - Create a translation with specific offsets
    /// * [`One::one`] - Trait implementation that returns identity
    #[inline]
    pub fn identity() -> Translation<T, D>
    where
        T: Zero,
    {
        Self::from(SVector::<T, D>::from_element(T::zero()))
    }

    /// Converts the scalar type of this translation's components to another type.
    ///
    /// This method allows you to convert between different numeric types, such as from
    /// `f64` to `f32`, or from `f32` to `f64`. This is useful when interfacing with
    /// APIs that require specific precision or when optimizing for performance vs. accuracy.
    ///
    /// # Type Parameters
    ///
    /// * `To` - The target scalar type for the translation components
    ///
    /// # Returns
    ///
    /// A new translation with the same geometric meaning but with components of type `To`.
    ///
    /// # Examples
    ///
    /// Converting from double to single precision:
    /// ```
    /// # use nalgebra::Translation2;
    /// let t_f64 = Translation2::new(1.0f64, 2.0);
    /// let t_f32 = t_f64.cast::<f32>();
    /// assert_eq!(t_f32, Translation2::new(1.0f32, 2.0));
    /// ```
    ///
    /// Converting from single to double precision:
    /// ```
    /// # use nalgebra::Translation3;
    /// let t_f32 = Translation3::new(1.5f32, 2.5, 3.5);
    /// let t_f64 = t_f32.cast::<f64>();
    /// assert_eq!(t_f64, Translation3::new(1.5f64, 2.5, 3.5));
    /// ```
    ///
    /// Practical use - interfacing with a graphics API:
    /// ```
    /// # use nalgebra::Translation2;
    /// // Calculate in high precision
    /// let precise_offset = Translation2::new(1.0f64 / 3.0, 2.0 / 3.0);
    ///
    /// // Convert to GPU-friendly format
    /// let gpu_offset = precise_offset.cast::<f32>();
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Translation::from`] - Create translations from other types
    pub fn cast<To: Scalar>(self) -> Translation<To, D>
    where
        Translation<To, D>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T: Scalar + Zero + ClosedAddAssign, const D: usize> One for Translation<T, D> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: Scalar, const D: usize> Distribution<Translation<T, D>> for StandardUniform
where
    StandardUniform: Distribution<T>,
{
    /// Generate an arbitrary random variate for testing purposes.
    #[inline]
    fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> Translation<T, D> {
        Translation::from(rng.random::<SVector<T, D>>())
    }
}

#[cfg(feature = "arbitrary")]
impl<T: Scalar + Arbitrary + Send, const D: usize> Arbitrary for Translation<T, D>
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
 * Small translation construction from components.
 *
 */
macro_rules! componentwise_constructors_impl(
    ($($doc: expr_2021; $D: expr_2021, $($args: ident:$irow: expr_2021),*);* $(;)*) => {$(
        impl<T> Translation<T, $D>
             {
            #[doc = "Creates a new translation from its individual components.\n\n"]
            #[doc = "A translation represents a shift or displacement in space. This constructor\n"]
            #[doc = "allows you to specify the displacement amount in each dimension directly.\n\n"]
            #[doc = "# Arguments\n\n"]
            #[doc = "Each argument represents the displacement in one dimension:\n"]
            #[doc = "- For 2D: `x` (horizontal), `y` (vertical)\n"]
            #[doc = "- For 3D: `x` (horizontal), `y` (vertical), `z` (depth)\n\n"]
            #[doc = "Positive values move in the positive direction of that axis.\n\n"]
            #[doc = "# Examples\n\n"]
            #[doc = "Basic usage:\n```"]
            #[doc = $doc]
            #[doc = "```\n\n"]
            #[doc = "# See Also\n\n"]
            #[doc = "* [`Translation::identity`] - Create a translation with zero offset\n"]
            #[doc = "* [`Translation::from`] - Create from a vector or other types"]
            #[inline]
            pub const fn new($($args: T),*) -> Self {
                Self { vector: SVector::<T, $D>::new($($args),*) }
            }
        }
    )*}
);

componentwise_constructors_impl!(
    "# use nalgebra::Translation1;\nlet t = Translation1::new(1.0);\nassert!(t.vector.x == 1.0);";
    1, x:0;
    "# use nalgebra::Translation2;\nlet t = Translation2::new(1.0, 2.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0);";
    2, x:0, y:1;
    "# use nalgebra::Translation3;\nlet t = Translation3::new(1.0, 2.0, 3.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0 && t.vector.z == 3.0);";
    3, x:0, y:1, z:2;
    "# use nalgebra::Translation4;\nlet t = Translation4::new(1.0, 2.0, 3.0, 4.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0 && t.vector.z == 3.0 && t.vector.w == 4.0);";
    4, x:0, y:1, z:2, w:3;
    "# use nalgebra::Translation5;\nlet t = Translation5::new(1.0, 2.0, 3.0, 4.0, 5.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0 && t.vector.z == 3.0 && t.vector.w == 4.0 && t.vector.a == 5.0);";
    5, x:0, y:1, z:2, w:3, a:4;
    "# use nalgebra::Translation6;\nlet t = Translation6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);\nassert!(t.vector.x == 1.0 && t.vector.y == 2.0 && t.vector.z == 3.0 && t.vector.w == 4.0 && t.vector.a == 5.0 && t.vector.b == 6.0);";
    6, x:0, y:1, z:2, w:3, a:4, b:5;
);
