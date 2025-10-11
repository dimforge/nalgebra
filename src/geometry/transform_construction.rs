use num::One;

use simba::scalar::RealField;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, OMatrix};

use crate::geometry::{TCategory, Transform};

impl<T: RealField, C: TCategory, const D: usize> Default for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: RealField, C: TCategory, const D: usize> Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    /// Creates a new identity transformation.
    ///
    /// The identity transformation leaves all points and vectors unchanged. It's represented by
    /// the identity matrix, which has 1s on the diagonal and 0s elsewhere. This is the neutral
    /// element for transformation composition: `T * identity = identity * T = T`.
    ///
    /// # Returns
    ///
    /// An identity transformation that performs no change
    ///
    /// # Example: 2D Identity
    ///
    /// ```
    /// # use nalgebra::{Transform2, Projective2, Affine2, Point2};
    /// # use approx::assert_relative_eq;
    /// let pt = Point2::new(1.0, 2.0);
    ///
    /// // All transform types can create identity
    /// let gen_id = Transform2::identity();
    /// assert_eq!(gen_id * pt, pt);
    ///
    /// let proj_id = Projective2::identity();
    /// assert_eq!(proj_id * pt, pt);
    ///
    /// let aff_id = Affine2::identity();
    /// assert_eq!(aff_id * pt, pt);
    /// ```
    ///
    /// # Example: 3D Identity
    ///
    /// ```
    /// # use nalgebra::{Transform3, Projective3, Affine3, Point3, Vector3};
    /// let pt = Point3::new(1.0, 2.0, 3.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    ///
    /// let id = Projective3::identity();
    /// assert_eq!(id * pt, pt);
    /// assert_eq!(id * vec, vec);
    /// ```
    ///
    /// # Example: Composition with Identity
    ///
    /// ```
    /// # use nalgebra::{Affine2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// // Create a scaling transform
    /// let scale = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 3.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// );
    /// let t = Affine2::from_matrix_unchecked(scale);
    ///
    /// // Composing with identity doesn't change the transform
    /// let id = Affine2::identity();
    /// assert_relative_eq!(t * id, t);
    /// assert_relative_eq!(id * t, t);
    /// ```
    ///
    /// # Example: Reset Transform
    ///
    /// ```
    /// # use nalgebra::{Transform3, Point3};
    /// // Start with identity, then modify later
    /// let mut transform = Transform3::identity();
    ///
    /// // ... apply various transformations ...
    ///
    /// // Reset to identity
    /// transform = Transform3::identity();
    ///
    /// // Now it's back to no transformation
    /// let pt = Point3::new(1.0, 2.0, 3.0);
    /// assert_eq!(transform.transform_point(&pt), pt);
    /// ```
    ///
    /// # Example: Building Transform Chains
    ///
    /// ```
    /// # use nalgebra::{Projective2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// // Start with identity
    /// let mut combined = Projective2::identity();
    ///
    /// // Chain transformations by multiplication
    /// let scale = Projective2::from_matrix_unchecked(Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// ));
    ///
    /// combined = scale * combined;  // Apply scaling
    /// // combined is now just the scaling transform
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_matrix_unchecked`](Self::from_matrix_unchecked) - Create from custom matrix
    /// * `Default` trait implementation - Also creates identity
    #[inline]
    pub fn identity() -> Self {
        Self::from_matrix_unchecked(OMatrix::<
            _,
            DimNameSum<Const<D>, U1>,
            DimNameSum<Const<D>, U1>,
        >::identity())
    }
}

impl<T: RealField, C: TCategory, const D: usize> One for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    /// Creates a new identity transform.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}
