use crate::storage::Storage;
use crate::{
    Allocator, DefaultAllocator, Dim, OVector, One, RealField, Scalar, Unit, Vector, Zero,
};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign, ClosedSubAssign};

/// # Interpolation
impl<
    T: Scalar + Zero + One + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign,
    D: Dim,
    S: Storage<T, D>,
> Vector<T, D, S>
{
    /// Returns `self * (1.0 - t) + rhs * t`, i.e., the linear blend of the vectors x and y using the scalar value a.
    ///
    /// The value for a is not restricted to the range `[0, 1]`.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let x = Vector3::new(1.0, 2.0, 3.0);
    /// let y = Vector3::new(10.0, 20.0, 30.0);
    /// assert_eq!(x.lerp(&y, 0.1), Vector3::new(1.9, 3.8, 5.7));
    /// ```
    #[must_use]
    pub fn lerp<S2: Storage<T, D>>(&self, rhs: &Vector<T, D, S2>, t: T) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        let mut res = self.clone_owned();
        res.axpy(t.clone(), rhs, T::one() - t);
        res
    }

    /// Computes the spherical linear interpolation between two non-zero vectors.
    ///
    /// The result is a unit vector.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector2};
    ///
    /// let v1 =Vector2::new(1.0, 2.0);
    /// let v2 = Vector2::new(2.0, -3.0);
    ///
    /// let v = v1.slerp(&v2, 1.0);
    ///
    /// assert_eq!(v, v2.normalize());
    /// ```
    #[must_use]
    pub fn slerp<S2: Storage<T, D>>(&self, rhs: &Vector<T, D, S2>, t: T) -> OVector<T, D>
    where
        T: RealField,
        DefaultAllocator: Allocator<D>,
    {
        let me = Unit::new_normalize(self.clone_owned());
        let rhs = Unit::new_normalize(rhs.clone_owned());
        me.slerp(&rhs, t).into_inner()
    }
}

/// # Interpolation between two unit vectors
impl<T: RealField, D: Dim, S: Storage<T, D>> Unit<Vector<T, D, S>> {
    /// Computes the spherical linear interpolation between two unit vectors.
    ///
    /// When the vectors are antiparallel the geodesic is ambiguous; an arbitrary but
    /// deterministic one is used, still honoring `slerp(_, 0) == self` and `slerp(_, 1) == rhs`.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector2};
    ///
    /// let v1 = Unit::new_normalize(Vector2::new(1.0, 2.0));
    /// let v2 = Unit::new_normalize(Vector2::new(2.0, -3.0));
    ///
    /// let v = v1.slerp(&v2, 1.0);
    ///
    /// assert_eq!(v, v2);
    /// ```
    #[must_use]
    pub fn slerp<S2: Storage<T, D>>(
        &self,
        rhs: &Unit<Vector<T, D, S2>>,
        t: T,
    ) -> Unit<OVector<T, D>>
    where
        DefaultAllocator: Allocator<D>,
    {
        if let Some(result) = self.try_slerp(rhs, t.clone(), T::default_epsilon()) {
            return result;
        }

        // `self` and `rhs` are (nearly) antiparallel: the great circle is ambiguous, but the
        // endpoints are not. Rotate through a deterministic axis orthogonal to `self`, so
        // slerp(_, 0) == self, slerp(_, 1) == rhs, and the path stays continuous. See #657.
        let n = self.clone_owned();
        let dim = n.len();

        // Canonical basis vector least aligned with `n`, for numerical robustness.
        let mut axis = 0;
        let mut min_sq = n[0].clone() * n[0].clone();
        for i in 1..dim {
            let sq = n[i].clone() * n[i].clone();
            if sq < min_sq {
                min_sq = sq;
                axis = i;
            }
        }

        let mut ortho = n.clone();
        ortho.fill(T::zero());
        ortho[axis] = T::one();
        let dot = ortho.dot(&n);
        ortho.axpy(-dot, &n, T::one()); // ortho = e_axis - (e_axis . n) n

        let ortho_norm = ortho.norm();
        if relative_eq!(ortho_norm, T::zero()) {
            // No orthogonal direction exists (1D): only the endpoints are defined.
            let half = T::one() / (T::one() + T::one());
            return if t <= half {
                Unit::new_unchecked(n)
            } else {
                Unit::new_unchecked(rhs.clone_owned())
            };
        }
        ortho.unscale_mut(ortho_norm);

        let theta = T::pi() * t;
        let mut res = n.scale(theta.clone().cos());
        res.axpy(theta.sin(), &ortho, T::one());

        Unit::new_unchecked(res)
    }

    /// Computes the spherical linear interpolation between two unit vectors.
    ///
    /// Returns `None` if the two vectors are almost collinear and with opposite direction
    /// (in this case, there is an infinity of possible results).
    #[must_use]
    pub fn try_slerp<S2: Storage<T, D>>(
        &self,
        rhs: &Unit<Vector<T, D, S2>>,
        t: T,
        epsilon: T,
    ) -> Option<Unit<OVector<T, D>>>
    where
        DefaultAllocator: Allocator<D>,
    {
        let c_hang = self.dot(rhs);

        // self == other
        if c_hang >= T::one() {
            return Some(Unit::new_unchecked(self.clone_owned()));
        }

        // self == -other, up to rounding pushing the dot product past -1 (which would make the
        // acos/sqrt below NaN): opposite direction, so the interpolation is not well-defined.
        if c_hang <= -T::one() {
            return None;
        }

        let hang = c_hang.clone().acos();
        let s_hang = (T::one() - c_hang.clone() * c_hang).sqrt();

        // TODO: what if s_hang is 0.0 ? The result is not well-defined.
        if relative_eq!(s_hang, T::zero(), epsilon = epsilon) {
            None
        } else {
            let ta = ((T::one() - t.clone()) * hang.clone()).sin() / s_hang.clone();
            let tb = (t * hang).sin() / s_hang;
            let mut res = self.scale(ta);
            res.axpy(tb, &**rhs, T::one());

            Some(Unit::new_unchecked(res))
        }
    }
}
