#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use num::Zero;
use std::ops::Neg;

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Dim, DimName, Matrix, Normed, OMatrix, OVector};
use crate::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::storage::{Storage, StorageMut};
use crate::{ComplexField, Scalar, SimdComplexField, Unit};
use simba::scalar::ClosedNeg;
use simba::simd::{SimdOption, SimdPartialOrd, SimdValue};

// TODO: this should be be a trait on alga?
/// A trait for abstract matrix norms.
///
/// This may be moved to the alga crate in the future.
pub trait Norm<T: SimdComplexField> {
    /// Apply this norm to the given matrix.
    fn norm<R, C, S>(&self, m: &Matrix<T, R, C, S>) -> T::SimdRealField
    where
        R: Dim,
        C: Dim,
        S: Storage<T, R, C>;
    /// Use the metric induced by this norm to compute the metric distance between the two given matrices.
    fn metric_distance<R1, C1, S1, R2, C2, S2>(
        &self,
        m1: &Matrix<T, R1, C1, S1>,
        m2: &Matrix<T, R2, C2, S2>,
    ) -> T::SimdRealField
    where
        R1: Dim,
        C1: Dim,
        S1: Storage<T, R1, C1>,
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>;
}

/// Euclidean norm.
#[derive(Copy, Clone, Debug)]
pub struct EuclideanNorm;
/// Lp norm.
#[derive(Copy, Clone, Debug)]
pub struct LpNorm(pub i32);
/// L-infinite norm aka. Chebytchev norm aka. uniform norm aka. suppremum norm.
#[derive(Copy, Clone, Debug)]
pub struct UniformNorm;

impl<T: SimdComplexField> Norm<T> for EuclideanNorm {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<T, R, C, S>) -> T::SimdRealField
    where
        R: Dim,
        C: Dim,
        S: Storage<T, R, C>,
    {
        m.norm_squared().simd_sqrt()
    }

    #[inline]
    fn metric_distance<R1, C1, S1, R2, C2, S2>(
        &self,
        m1: &Matrix<T, R1, C1, S1>,
        m2: &Matrix<T, R2, C2, S2>,
    ) -> T::SimdRealField
    where
        R1: Dim,
        C1: Dim,
        S1: Storage<T, R1, C1>,
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        m1.zip_fold(m2, T::SimdRealField::zero(), |acc, a, b| {
            let diff = a - b;
            acc + diff.simd_modulus_squared()
        })
        .simd_sqrt()
    }
}

impl<T: SimdComplexField> Norm<T> for LpNorm {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<T, R, C, S>) -> T::SimdRealField
    where
        R: Dim,
        C: Dim,
        S: Storage<T, R, C>,
    {
        m.fold(T::SimdRealField::zero(), |a, b| {
            a + b.simd_modulus().simd_powi(self.0)
        })
        .simd_powf(crate::convert(1.0 / (self.0 as f64)))
    }

    #[inline]
    fn metric_distance<R1, C1, S1, R2, C2, S2>(
        &self,
        m1: &Matrix<T, R1, C1, S1>,
        m2: &Matrix<T, R2, C2, S2>,
    ) -> T::SimdRealField
    where
        R1: Dim,
        C1: Dim,
        S1: Storage<T, R1, C1>,
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        m1.zip_fold(m2, T::SimdRealField::zero(), |acc, a, b| {
            let diff = a - b;
            acc + diff.simd_modulus().simd_powi(self.0)
        })
        .simd_powf(crate::convert(1.0 / (self.0 as f64)))
    }
}

impl<T: SimdComplexField> Norm<T> for UniformNorm {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<T, R, C, S>) -> T::SimdRealField
    where
        R: Dim,
        C: Dim,
        S: Storage<T, R, C>,
    {
        // NOTE: we don't use `m.amax()` here because for the complex
        // numbers this will return the max norm1 instead of the modulus.
        m.fold(T::SimdRealField::zero(), |acc, a| {
            acc.simd_max(a.simd_modulus())
        })
    }

    #[inline]
    fn metric_distance<R1, C1, S1, R2, C2, S2>(
        &self,
        m1: &Matrix<T, R1, C1, S1>,
        m2: &Matrix<T, R2, C2, S2>,
    ) -> T::SimdRealField
    where
        R1: Dim,
        C1: Dim,
        S1: Storage<T, R1, C1>,
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        m1.zip_fold(m2, T::SimdRealField::zero(), |acc, a, b| {
            let val = (a - b).simd_modulus();
            acc.simd_max(val)
        })
    }
}

/// # Magnitude and norms
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Computes the squared L2 norm (Euclidean norm) of this matrix or vector.
    ///
    /// This is more efficient than [`norm()`](Self::norm) since it avoids the square root computation.
    /// It's useful when you only need to compare magnitudes or when the actual norm value isn't needed.
    ///
    /// For a vector, this computes the sum of the squares of all components.
    /// For a matrix, this computes the Frobenius norm squared.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{Vector3, Matrix2};
    ///
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// // 1² + 2² + 3² = 14
    /// assert_eq!(v.norm_squared(), 14.0);
    ///
    /// // Useful for comparing distances without sqrt
    /// let v1 = Vector3::new(1.0, 0.0, 0.0);
    /// let v2 = Vector3::new(2.0, 0.0, 0.0);
    /// let v3 = Vector3::new(3.0, 0.0, 0.0);
    /// assert!(v1.norm_squared() < v2.norm_squared());
    /// assert!(v2.norm_squared() < v3.norm_squared());
    ///
    /// // Matrix example
    /// let m = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    /// assert_eq!(m.norm_squared(), 30.0);  // 1² + 2² + 3² + 4² = 30
    /// ```
    #[inline]
    #[must_use]
    pub fn norm_squared(&self) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        let mut res = T::SimdRealField::zero();

        for i in 0..self.ncols() {
            let col = self.column(i);
            res += col.dotc(&col).simd_real()
        }

        res
    }

    /// Computes the L2 norm (Euclidean norm / magnitude / length) of this matrix or vector.
    ///
    /// For vectors, this is the standard Euclidean length: `sqrt(x² + y² + z² + ...)`.
    /// For matrices, this computes the Frobenius norm (square root of sum of squared elements).
    ///
    /// This is equivalent to [`magnitude()`](Self::magnitude).
    /// Use [`.apply_norm()`](Self::apply_norm) to apply a custom norm (L1, L-infinity, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{Vector2, Vector3, Matrix2};
    ///
    /// // 2D vector: Pythagorean theorem
    /// let v = Vector2::new(3.0, 4.0);
    /// assert_eq!(v.norm(), 5.0);  // 3-4-5 triangle
    ///
    /// // 3D vector
    /// let v = Vector3::new(1.0, 2.0, 2.0);
    /// assert_eq!(v.norm(), 3.0);  // sqrt(1 + 4 + 4) = 3
    ///
    /// // Unit vectors have norm 1
    /// let unit_x = Vector3::x();
    /// assert_eq!(unit_x.norm(), 1.0);
    ///
    /// // Matrix Frobenius norm
    /// let m = Matrix2::new(1.0, 0.0,
    ///                      0.0, 1.0);
    /// assert!((m.norm() - 1.414213).abs() < 1e-5);  // sqrt(2)
    /// ```
    ///
    /// # See Also
    ///
    /// * [`norm_squared()`](Self::norm_squared) - More efficient when you don't need the actual value
    /// * [`magnitude()`](Self::magnitude) - Alias for `norm()`
    /// * [`normalize()`](Self::normalize) - Get a unit vector in the same direction
    #[inline]
    #[must_use]
    pub fn norm(&self) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        self.norm_squared().simd_sqrt()
    }

    /// Compute the distance between `self` and `rhs` using the metric induced by the euclidean norm.
    ///
    /// Use `.apply_metric_distance` to apply a custom norm.
    #[inline]
    #[must_use]
    pub fn metric_distance<R2, C2, S2>(&self, rhs: &Matrix<T, R2, C2, S2>) -> T::SimdRealField
    where
        T: SimdComplexField,
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        self.apply_metric_distance(rhs, &EuclideanNorm)
    }

    /// Uses the given `norm` to compute the norm of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Vector3, UniformNorm, LpNorm, EuclideanNorm};
    ///
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// assert_eq!(v.apply_norm(&UniformNorm), 3.0);
    /// assert_eq!(v.apply_norm(&LpNorm(1)), 6.0);
    /// assert_eq!(v.apply_norm(&EuclideanNorm), v.norm());
    /// ```
    #[inline]
    #[must_use]
    pub fn apply_norm(&self, norm: &impl Norm<T>) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        norm.norm(self)
    }

    /// Uses the metric induced by the given `norm` to compute the metric distance between `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Vector3, UniformNorm, LpNorm, EuclideanNorm};
    ///
    /// let v1 = Vector3::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3::new(10.0, 20.0, 30.0);
    ///
    /// assert_eq!(v1.apply_metric_distance(&v2, &UniformNorm), 27.0);
    /// assert_eq!(v1.apply_metric_distance(&v2, &LpNorm(1)), 27.0 + 18.0 + 9.0);
    /// assert_eq!(v1.apply_metric_distance(&v2, &EuclideanNorm), (v1 - v2).norm());
    /// ```
    #[inline]
    #[must_use]
    pub fn apply_metric_distance<R2, C2, S2>(
        &self,
        rhs: &Matrix<T, R2, C2, S2>,
        norm: &impl Norm<T>,
    ) -> T::SimdRealField
    where
        T: SimdComplexField,
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        norm.metric_distance(self, rhs)
    }

    /// A synonym for the norm of this matrix.
    ///
    /// Aka the length.
    ///
    /// This function is simply implemented as a call to `norm()`
    #[inline]
    #[must_use]
    pub fn magnitude(&self) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        self.norm()
    }

    /// A synonym for the squared norm of this matrix.
    ///
    /// Aka the squared length.
    ///
    /// This function is simply implemented as a call to `norm_squared()`
    #[inline]
    #[must_use]
    pub fn magnitude_squared(&self) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        self.norm_squared()
    }

    /// Sets the magnitude of this vector.
    #[inline]
    pub fn set_magnitude(&mut self, magnitude: T::SimdRealField)
    where
        T: SimdComplexField,
        S: StorageMut<T, R, C>,
    {
        let n = self.norm();
        self.scale_mut(magnitude / n)
    }

    /// Returns a normalized (unit length) version of this vector.
    ///
    /// A normalized vector has the same direction but a magnitude (length) of 1.
    /// This is also called a "unit vector". This is useful in computer graphics
    /// and physics when you need a direction but not a specific magnitude.
    ///
    /// This method returns a new vector; see [`normalize_mut()`](Self::normalize_mut) for in-place normalization.
    ///
    /// # Panics
    ///
    /// Panics if the norm is zero (you cannot normalize a zero vector).
    /// Use [`try_normalize()`](Self::try_normalize) for a safe alternative.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{Vector2, Vector3};
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let normalized = v.normalize();
    /// // Direction preserved, but length is now 1
    /// assert_eq!(normalized, Vector2::new(0.6, 0.8));
    /// assert!((normalized.norm() - 1.0).abs() < 1e-10);
    ///
    /// // Common use: get direction between two points
    /// let from = Vector3::new(0.0, 0.0, 0.0);
    /// let to = Vector3::new(10.0, 0.0, 0.0);
    /// let direction = (to - from).normalize();
    /// assert_eq!(direction, Vector3::x());  // Points along +X axis
    ///
    /// // Original vector unchanged (this isn't in-place)
    /// assert_eq!(v.norm(), 5.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`normalize_mut()`](Self::normalize_mut) - In-place version
    /// * [`try_normalize()`](Self::try_normalize) - Safe version that returns `Option`
    /// * [`norm()`](Self::norm) - Get the length
    #[inline]
    #[must_use = "Did you mean to use normalize_mut()?"]
    pub fn normalize(&self) -> OMatrix<T, R, C>
    where
        T: SimdComplexField,
        DefaultAllocator: Allocator<R, C>,
    {
        self.unscale(self.norm())
    }

    /// The Lp norm of this matrix.
    #[inline]
    #[must_use]
    pub fn lp_norm(&self, p: i32) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        self.apply_norm(&LpNorm(p))
    }

    /// Attempts to normalize `self`.
    ///
    /// The components of this matrix can be SIMD types.
    #[inline]
    #[must_use = "Did you mean to use simd_try_normalize_mut()?"]
    pub fn simd_try_normalize(&self, min_norm: T::SimdRealField) -> SimdOption<OMatrix<T, R, C>>
    where
        T: SimdComplexField,
        T::Element: Scalar,
        DefaultAllocator: Allocator<R, C>,
    {
        let n = self.norm();
        let le = n.clone().simd_le(min_norm);
        let val = self.unscale(n);
        SimdOption::new(val, le)
    }

    /// Sets the magnitude of this vector unless it is smaller than `min_magnitude`.
    ///
    /// If `self.magnitude()` is smaller than `min_magnitude`, it will be left unchanged.
    /// Otherwise this is equivalent to: `*self = self.normalize() * magnitude`.
    #[inline]
    pub fn try_set_magnitude(&mut self, magnitude: T::RealField, min_magnitude: T::RealField)
    where
        T: ComplexField,
        S: StorageMut<T, R, C>,
    {
        let n = self.norm();

        if n > min_magnitude {
            self.scale_mut(magnitude / n)
        }
    }

    /// Returns a new vector with the same magnitude as `self` clamped between `0.0` and `max`.
    #[inline]
    #[must_use]
    pub fn cap_magnitude(&self, max: T::RealField) -> OMatrix<T, R, C>
    where
        T: ComplexField,
        DefaultAllocator: Allocator<R, C>,
    {
        let n = self.norm();

        if n > max {
            self.scale(max / n)
        } else {
            self.clone_owned()
        }
    }

    /// Returns a new vector with the same magnitude as `self` clamped between `0.0` and `max`.
    #[inline]
    #[must_use]
    pub fn simd_cap_magnitude(&self, max: T::SimdRealField) -> OMatrix<T, R, C>
    where
        T: SimdComplexField,
        T::Element: Scalar,
        DefaultAllocator: Allocator<R, C>,
    {
        let n = self.norm();
        let scaled = self.scale(max.clone() / n.clone());
        let use_scaled = n.simd_gt(max);
        scaled.select(use_scaled, self.clone_owned())
    }

    /// Attempts to normalize this vector, returning `None` if the norm is too small.
    ///
    /// This is the safe version of [`normalize()`](Self::normalize) that returns `None` instead
    /// of panicking or producing invalid results when the vector is too small to normalize.
    ///
    /// A vector with a norm smaller than or equal to `min_norm` cannot be meaningfully normalized,
    /// so this function returns `None` in that case.
    ///
    /// # Arguments
    ///
    /// * `min_norm` - The minimum acceptable norm. Vectors with smaller norms return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Vector3;
    ///
    /// // Normal vector normalizes successfully
    /// let v = Vector3::new(3.0, 4.0, 0.0);
    /// let normalized = v.try_normalize(1e-6);
    /// assert!(normalized.is_some());
    /// assert!((normalized.unwrap().norm() - 1.0).abs() < 1e-10);
    ///
    /// // Zero vector cannot be normalized
    /// let zero = Vector3::new(0.0, 0.0, 0.0);
    /// assert_eq!(zero.try_normalize(1e-6), None);
    ///
    /// // Very small vector also returns None
    /// let tiny = Vector3::new(1e-10, 1e-10, 1e-10);
    /// assert_eq!(tiny.try_normalize(1e-6), None);
    ///
    /// // Use this for safety when normalizing user input
    /// let user_vector = Vector3::new(1.0, 0.0, 0.0);
    /// if let Some(dir) = user_vector.try_normalize(1e-6) {
    ///     // Safe to use as a direction
    ///     println!("Direction: {}", dir);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`normalize()`](Self::normalize) - Panics if norm is zero
    /// * [`try_normalize_mut()`](Self::try_normalize_mut) - In-place version
    #[inline]
    #[must_use = "Did you mean to use try_normalize_mut()?"]
    pub fn try_normalize(&self, min_norm: T::RealField) -> Option<OMatrix<T, R, C>>
    where
        T: ComplexField,
        DefaultAllocator: Allocator<R, C>,
    {
        let n = self.norm();

        if n <= min_norm {
            None
        } else {
            Some(self.unscale(n))
        }
    }
}

/// # In-place normalization
impl<T: Scalar, R: Dim, C: Dim, S: StorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Normalizes this vector in-place (modifying it) and returns its original norm.
    ///
    /// This is the in-place version of [`normalize()`](Self::normalize). It modifies
    /// the vector directly instead of creating a new one, which can be more efficient.
    ///
    /// After calling this method, the vector will have a magnitude of 1 (be a unit vector).
    /// The return value is the norm the vector had before normalization.
    ///
    /// # Panics
    ///
    /// Panics if the norm is zero. Use [`try_normalize_mut()`](Self::try_normalize_mut) for a safe alternative.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Vector3;
    ///
    /// let mut v = Vector3::new(3.0, 4.0, 0.0);
    /// let old_norm = v.normalize_mut();
    ///
    /// // Vector is now normalized
    /// assert_eq!(old_norm, 5.0);
    /// assert!((v.norm() - 1.0).abs() < 1e-10);
    /// assert_eq!(v, Vector3::new(0.6, 0.8, 0.0));
    ///
    /// // In-place normalization is more efficient than creating a new vector
    /// let mut direction = Vector3::new(1.0, 1.0, 1.0);
    /// direction.normalize_mut();
    /// // direction is now a unit vector
    /// ```
    ///
    /// # See Also
    ///
    /// * [`normalize()`](Self::normalize) - Returns a new normalized vector
    /// * [`try_normalize_mut()`](Self::try_normalize_mut) - Safe version that returns `Option`
    #[inline]
    pub fn normalize_mut(&mut self) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        let n = self.norm();
        self.unscale_mut(n.clone());

        n
    }

    /// Normalizes this matrix in-place and return its norm.
    ///
    /// The components of the matrix can be SIMD types.
    #[inline]
    #[must_use = "Did you mean to use simd_try_normalize_mut()?"]
    pub fn simd_try_normalize_mut(
        &mut self,
        min_norm: T::SimdRealField,
    ) -> SimdOption<T::SimdRealField>
    where
        T: SimdComplexField,
        T::Element: Scalar,
        DefaultAllocator: Allocator<R, C>,
    {
        let n = self.norm();
        let le = n.clone().simd_le(min_norm);
        self.apply(|e| *e = e.clone().simd_unscale(n.clone()).select(le, e.clone()));
        SimdOption::new(n, le)
    }

    /// Normalizes this matrix in-place or does nothing if its norm is smaller or equal to `eps`.
    ///
    /// If the normalization succeeded, returns the old norm of this matrix.
    #[inline]
    pub fn try_normalize_mut(&mut self, min_norm: T::RealField) -> Option<T::RealField>
    where
        T: ComplexField,
    {
        let n = self.norm();

        if n <= min_norm {
            None
        } else {
            self.unscale_mut(n.clone());
            Some(n)
        }
    }
}

impl<T: SimdComplexField, R: Dim, C: Dim> Normed for OMatrix<T, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Norm = T::SimdRealField;

    #[inline]
    fn norm(&self) -> T::SimdRealField {
        self.norm()
    }

    #[inline]
    fn norm_squared(&self) -> T::SimdRealField {
        self.norm_squared()
    }

    #[inline]
    fn scale_mut(&mut self, n: Self::Norm) {
        self.scale_mut(n)
    }

    #[inline]
    fn unscale_mut(&mut self, n: Self::Norm) {
        self.unscale_mut(n)
    }
}

impl<T: Scalar + ClosedNeg, R: Dim, C: Dim> Neg for Unit<OMatrix<T, R, C>>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Output = Unit<OMatrix<T, R, C>>;

    #[inline]
    fn neg(self) -> Self::Output {
        Unit::new_unchecked(-self.value)
    }
}

// TODO: specialization will greatly simplify this implementation in the future.
// In particular:
//   − use `x()` instead of `::canonical_basis_element`
//   − use `::new(x, y, z)` instead of `::from_slice`
/// # Basis and orthogonalization
impl<T: ComplexField, D: DimName> OVector<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// The i-the canonical basis element.
    #[inline]
    fn canonical_basis_element(i: usize) -> Self {
        let mut res = Self::zero();
        res[i] = T::one();
        res
    }

    /// Orthonormalizes the given family of vectors. The largest free family of vectors is moved at
    /// the beginning of the array and its size is returned. Vectors at an indices larger or equal to
    /// this length can be modified to an arbitrary value.
    #[inline]
    pub fn orthonormalize(vs: &mut [Self]) -> usize {
        let mut nbasis_elements = 0;

        for i in 0..vs.len() {
            {
                let (elt, basis) = vs[..i + 1].split_last_mut().unwrap();

                for basis_element in &basis[..nbasis_elements] {
                    *elt -= basis_element * elt.dot(basis_element)
                }
            }

            if vs[i].try_normalize_mut(T::RealField::zero()).is_some() {
                // TODO: this will be efficient on dynamically-allocated vectors but for
                // statically-allocated ones, `.clone_from` would be better.
                vs.swap(nbasis_elements, i);
                nbasis_elements += 1;

                // All the other vectors will be dependent.
                if nbasis_elements == D::DIM {
                    break;
                }
            }
        }

        nbasis_elements
    }

    /// Applies the given closure to each element of the orthonormal basis of the subspace
    /// orthogonal to free family of vectors `vs`. If `vs` is not a free family, the result is
    /// unspecified.
    // TODO: return an iterator instead when `-> impl Iterator` will be supported by Rust.
    #[inline]
    pub fn orthonormal_subspace_basis<F>(vs: &[Self], mut f: F)
    where
        F: FnMut(&Self) -> bool,
    {
        // TODO: is this necessary?
        assert!(
            vs.len() <= D::DIM,
            "The given set of vectors has no chance of being a free family."
        );

        match D::DIM {
            1 => {
                if vs.is_empty() {
                    let _ = f(&Self::canonical_basis_element(0));
                }
            }
            2 => {
                if vs.is_empty() {
                    let _ = f(&Self::canonical_basis_element(0))
                        && f(&Self::canonical_basis_element(1));
                } else if vs.len() == 1 {
                    let v = &vs[0];
                    let res = Self::from_column_slice(&[-v[1].clone(), v[0].clone()]);

                    let _ = f(&res.normalize());
                }

                // Otherwise, nothing.
            }
            3 => {
                if vs.is_empty() {
                    let _ = f(&Self::canonical_basis_element(0))
                        && f(&Self::canonical_basis_element(1))
                        && f(&Self::canonical_basis_element(2));
                } else if vs.len() == 1 {
                    let v = &vs[0];
                    let mut a;

                    if v[0].clone().norm1() > v[1].clone().norm1() {
                        a = Self::from_column_slice(&[v[2].clone(), T::zero(), -v[0].clone()]);
                    } else {
                        a = Self::from_column_slice(&[T::zero(), -v[2].clone(), v[1].clone()]);
                    };

                    let _ = a.normalize_mut();

                    if f(&a.cross(v)) {
                        let _ = f(&a);
                    }
                } else if vs.len() == 2 {
                    let _ = f(&vs[0].cross(&vs[1]).normalize());
                }
            }
            _ => {
                #[cfg(any(feature = "std", feature = "alloc"))]
                {
                    // XXX: use a GenericArray instead.
                    let mut known_basis = Vec::new();

                    for v in vs.iter() {
                        known_basis.push(v.normalize())
                    }

                    for i in 0..D::DIM - vs.len() {
                        let mut elt = Self::canonical_basis_element(i);

                        for v in &known_basis {
                            elt -= v * elt.dot(v)
                        }

                        if let Some(subsp_elt) = elt.try_normalize(T::RealField::zero()) {
                            if !f(&subsp_elt) {
                                return;
                            };

                            known_basis.push(subsp_elt);
                        }
                    }
                }
                #[cfg(all(not(feature = "std"), not(feature = "alloc")))]
                {
                    panic!(
                        "Cannot compute the orthogonal subspace basis of a vector with a dimension greater than 3 \
                            if #![no_std] is enabled and the 'alloc' feature is not enabled."
                    )
                }
            }
        }
    }
}
