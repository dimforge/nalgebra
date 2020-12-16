use crate::{Quaternion, SimdRealField};

/// A dual quaternion.
///
/// # Indexing
///
/// DualQuaternions are stored as \[..real, ..dual\].
/// Both of the quaternion components are laid out in `w, i, j, k` order.
///
/// ```
/// # use nalgebra::{DualQuaternion, Quaternion};
///
/// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
/// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
///
/// let dq = DualQuaternion::from_real_and_dual(real, dual);
/// assert_eq!(dq[0], 1.0);
/// assert_eq!(dq[4], 5.0);
/// assert_eq!(dq[6], 7.0);
/// ```
///
/// NOTE:
///  As of December 2020, dual quaternion support is a work in progress.
///  If a feature that you need is missing, feel free to open an issue or a PR.
///  See https://github.com/dimforge/nalgebra/issues/487
#[repr(C)]
#[derive(Debug, Default, Eq, PartialEq, Copy, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct DualQuaternion<N: SimdRealField> {
    // [real(w, i, j, k), dual(w, i, j, k)]
    pub(crate) dq: [N; 8],
}

impl<N: SimdRealField> DualQuaternion<N> {
    /// Get the first quaternion component.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DualQuaternion, Quaternion};
    ///
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    ///
    /// let dq = DualQuaternion::from_real_and_dual(real, dual);
    /// relative_eq!(dq.real(), real);
    /// ```
    #[inline]
    pub fn real(&self) -> Quaternion<N> {
        Quaternion::new(self[0], self[1], self[2], self[3])
    }

    /// Get the second quaternion component.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DualQuaternion, Quaternion};
    ///
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    ///
    /// let dq = DualQuaternion::from_real_and_dual(real, dual);
    /// relative_eq!(dq.dual(), dual);
    /// ```
    #[inline]
    pub fn dual(&self) -> Quaternion<N> {
        Quaternion::new(self[4], self[5], self[6], self[7])
    }
}

impl<N: SimdRealField> DualQuaternion<N>
where
    N::Element: SimdRealField,
{
    /// Normalizes this quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let dq = DualQuaternion::from_real_and_dual(real, dual);
    ///
    /// let dq_normalized = dq.normalize();
    ///
    /// relative_eq!(dq_normalized.real().norm(), 1.0);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use normalize_mut()?"]
    pub fn normalize(&self) -> Self {
        let real_norm = self.real().norm();

        Self::from_real_and_dual(self.real() / real_norm, self.dual() / real_norm)
    }

    /// Normalizes this quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let mut dq = DualQuaternion::from_real_and_dual(real, dual);
    ///
    /// dq.normalize_mut();
    ///
    /// relative_eq!(dq.real().norm(), 1.0);
    /// ```
    #[inline]
    pub fn normalize_mut(&mut self) {
        *self = self.normalize();
    }
}
