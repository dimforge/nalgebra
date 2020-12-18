use crate::{Quaternion, SimdRealField};
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A dual quaternion.
///
/// # Indexing
///
/// DualQuaternions are stored as \[..real, ..dual\].
/// Both of the quaternion components are laid out in `i, j, k, w` order.
///
/// ```
/// # use nalgebra::{DualQuaternion, Quaternion};
///
/// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
/// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
///
/// let dq = DualQuaternion::from_real_and_dual(real, dual);
/// assert_eq!(dq[0], 2.0);
/// assert_eq!(dq[1], 3.0);
///
/// assert_eq!(dq[4], 6.0);
/// assert_eq!(dq[7], 5.0);
/// ```
///
/// NOTE:
///  As of December 2020, dual quaternion support is a work in progress.
///  If a feature that you need is missing, feel free to open an issue or a PR.
///  See https://github.com/dimforge/nalgebra/issues/487
#[repr(C)]
#[derive(Debug, Default, Eq, PartialEq, Copy, Clone)]
pub struct DualQuaternion<N: SimdRealField> {
    /// The real component of the quaternion
    pub real: Quaternion<N>,
    /// The dual component of the quaternion
    pub dual: Quaternion<N>,
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
    /// relative_eq!(dq_normalized.real.norm(), 1.0);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use normalize_mut()?"]
    pub fn normalize(&self) -> Self {
        let real_norm = self.real.norm();

        Self::from_real_and_dual(self.real / real_norm, self.dual / real_norm)
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
    /// relative_eq!(dq.real.norm(), 1.0);
    /// ```
    #[inline]
    pub fn normalize_mut(&mut self) {
        *self = self.normalize();
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: SimdRealField> Serialize for DualQuaternion<N>
where
    N: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        self.as_ref().serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: SimdRealField> Deserialize<'a> for DualQuaternion<N>
where
    N: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        type Dq<N> = [N; 8];

        let dq: Dq<N> = Dq::<N>::deserialize(deserializer)?;

        Ok(Self {
            real: Quaternion::new(dq[3], dq[0], dq[1], dq[2]),
            dual: Quaternion::new(dq[7], dq[4], dq[5], dq[6]),
        })
    }
}
