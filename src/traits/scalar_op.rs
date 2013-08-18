/**
 * Trait of objects having an addition with a scalar.
 */
pub trait ScalarAdd<N> {
    /// Gets the result of an addition by a scalar.
    fn scalar_add(&self, &N) -> Self;

    /// In-place version of `scalar_add`.
    fn scalar_add_inplace(&mut self, &N);
}

/**
 * Trait of objects having a subtraction with a scalar.
 */
pub trait ScalarSub<N> {
    /// Gets the result of a subtraction by a scalar.
    fn scalar_sub(&self, &N) -> Self;

    /// In-place version of `scalar_sub`.
    fn scalar_sub_inplace(&mut self, &N);
}
