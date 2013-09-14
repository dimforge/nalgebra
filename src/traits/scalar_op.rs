// XXX: those traits should not exist since there is generalized operator overloading of Add and
// Sub.
// However, using the same trait multiple time as a trait bound (ex: impl<T: Add<N, V> + Add<V, V>)
// does not work properly, mainly because the way we are doing generalized operator overloading is
// verry hacky.
//
// Hopefull, this will be fixed on the future.

/**
 * Trait of objects having an addition with a scalar.
 */
pub trait ScalarAdd<N> {
    /// Gets the result of an addition by a scalar.
    fn add_s(&self, &N) -> Self;
}

impl<N, T: Add<N, T>> ScalarAdd<N> for T {
    /// Gets the result of an addition by a scalar.
    fn add_s(&self, n: &N) -> T {
        *self + *n
    }
}

/**
 * Trait of objects having a subtraction with a scalar.
 */
pub trait ScalarSub<N> {
    /// Gets the result of a subtraction by a scalar.
    fn sub_s(&self, &N) -> Self;
}

impl<N, T: Sub<N, T>> ScalarSub<N> for T {
    /// Gets the result of an subition by a scalar.
    fn sub_s(&self, n: &N) -> T {
        *self - *n
    }
}
