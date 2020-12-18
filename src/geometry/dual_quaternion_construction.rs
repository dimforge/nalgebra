use crate::{DualQuaternion, Quaternion, SimdRealField};

impl<N: SimdRealField> DualQuaternion<N> {
    /// Creates a dual quaternion from its rotation and translation components.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let rot = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let trans = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    ///
    /// let dq = DualQuaternion::from_real_and_dual(rot, trans);
    /// assert_eq!(dq.real.w, 1.0);
    /// ```
    #[inline]
    pub fn from_real_and_dual(real: Quaternion<N>, dual: Quaternion<N>) -> Self {
        Self { real, dual }
    }
}

impl<N: SimdRealField> DualQuaternion<N> {
    /// The dual quaternion multiplicative identity
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    ///
    /// let dq1 = DualQuaternion::identity();
    /// let dq2 = DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(1.,2.,3.,4.),
    ///     Quaternion::new(5.,6.,7.,8.)
    /// );
    ///
    /// assert_eq!(dq1 * dq2, dq2);
    /// assert_eq!(dq2 * dq1, dq2);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::from_real_and_dual(
            Quaternion::from_real(N::one()),
            Quaternion::from_real(N::zero()),
        )
    }
}
