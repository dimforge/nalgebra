use crate::storage::Storage;
use crate::{
    Allocator, DefaultAllocator, Dim, One, RealField, Scalar, Unit, Vector, VectorN, Zero,
};
use simba::scalar::{ClosedAdd, ClosedMul, ClosedSub};

/// # Interpolation
impl<N: Scalar + Zero + One + ClosedAdd + ClosedSub + ClosedMul, D: Dim, S: Storage<N, D>>
    Vector<N, D, S>
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
    pub fn lerp<S2: Storage<N, D>>(&self, rhs: &Vector<N, D, S2>, t: N) -> VectorN<N, D>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        let mut res = self.clone_owned();
        res.axpy(t.inlined_clone(), rhs, N::one() - t);
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
    pub fn slerp<S2: Storage<N, D>>(&self, rhs: &Vector<N, D, S2>, t: N) -> VectorN<N, D>
    where
        N: RealField,
        DefaultAllocator: Allocator<N, D>,
    {
        let me = Unit::new_normalize(self.clone_owned());
        let rhs = Unit::new_normalize(rhs.clone_owned());
        me.slerp(&rhs, t).into_inner()
    }
}

/// # Interpolation between two unit vectors
impl<N: RealField, D: Dim, S: Storage<N, D>> Unit<Vector<N, D, S>> {
    /// Computes the spherical linear interpolation between two unit vectors.
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
    pub fn slerp<S2: Storage<N, D>>(
        &self,
        rhs: &Unit<Vector<N, D, S2>>,
        t: N,
    ) -> Unit<VectorN<N, D>>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        // TODO: the result is wrong when self and rhs are collinear with opposite direction.
        self.try_slerp(rhs, t, N::default_epsilon())
            .unwrap_or_else(|| Unit::new_unchecked(self.clone_owned()))
    }

    /// Computes the spherical linear interpolation between two unit vectors.
    ///
    /// Returns `None` if the two vectors are almost collinear and with opposite direction
    /// (in this case, there is an infinity of possible results).
    pub fn try_slerp<S2: Storage<N, D>>(
        &self,
        rhs: &Unit<Vector<N, D, S2>>,
        t: N,
        epsilon: N,
    ) -> Option<Unit<VectorN<N, D>>>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        let c_hang = self.dot(rhs);

        // self == other
        if c_hang >= N::one() {
            return Some(Unit::new_unchecked(self.clone_owned()));
        }

        let hang = c_hang.acos();
        let s_hang = (N::one() - c_hang * c_hang).sqrt();

        // TODO: what if s_hang is 0.0 ? The result is not well-defined.
        if relative_eq!(s_hang, N::zero(), epsilon = epsilon) {
            None
        } else {
            let ta = ((N::one() - t) * hang).sin() / s_hang;
            let tb = (t * hang).sin() / s_hang;
            let mut res = self.scale(ta);
            res.axpy(tb, &**rhs, N::one());

            Some(Unit::new_unchecked(res))
        }
    }
}
