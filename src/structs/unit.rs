use traits::geometry::Norm;


/// A wrapper that ensures the undelying algebraic entity has a unit norm.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Unit<T> {
    v: T
}

impl<T: Norm> Unit<T> {
    /// Normalize the given value and return it wrapped on a `Unit` structure.
    #[inline]
    pub fn new(v: &T) -> Self {
        Unit { v: v.normalize() }
    }

    /// Attempts to normalize the given value and return it wrapped on a `Unit` structure.
    ///
    /// Returns `None` if the norm was smaller or equal to `min_norm`.
    #[inline]
    pub fn try_new(v: &T, min_norm: T::NormType) -> Option<Self> {
        v.try_normalize(min_norm).map(|v| Unit { v: v })
    }

    /// Normalize the given value and return it wrapped on a `Unit` structure and its norm.
    #[inline]
    pub fn new_and_get(mut v: T) -> (Self, T::NormType) {
        let n = v.normalize_mut();

        (Unit { v: v }, n)
    }

    /// Normalize the given value and return it wrapped on a `Unit` structure and its norm.
    ///
    /// Returns `None` if the norm was smaller or equal to `min_norm`.
    #[inline]
    pub fn try_new_and_get(mut v: T, min_norm: T::NormType) -> Option<(Self, T::NormType)> {
        if let Some(n) = v.try_normalize_mut(min_norm) {
            Some((Unit { v: v }, n))
        }
        else {
            None
        }
    }

    /// Normalizes this value again. This is useful when repeated computations 
    /// might cause a drift in the norm because of float inaccuracies.
    ///
    /// Returns the norm beform re-normalization (should be close to `1.0`).
    #[inline]
    pub fn renormalize(&mut self) -> T::NormType {
        self.v.normalize_mut()
    }
}

impl<T> Unit<T> {
    /// Wraps the given value, assuming it is already normalized.
    ///
    /// This function is not safe because `v` is not verified to be actually normalized.
    #[inline]
    pub fn from_unit_value_unchecked(v: T) -> Self {
        Unit { v: v }
    }

    /// Retrieves the underlying value.
    #[inline]
    pub fn unwrap(self) -> T {
        self.v
    }
}

impl<T> AsRef<T> for Unit<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.v
    }
}
