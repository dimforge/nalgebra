/**
 * Trait of objects having a multiplication with a scalar.
 */
pub trait ScalarMul<N>
{
  /// Gets the result of a multiplication by a scalar.
  fn scalar_mul(&self, &N) -> Self;

  /// In-place version of `scalar_mul`.
  fn scalar_mul_inplace(&mut self, &N);
}

/**
 * Trait of objects having a division with a scalar.
 */
pub trait ScalarDiv<N>
{
  /// Gets the result of a division by a scalar.
  fn scalar_div(&self, &N) -> Self;

  /// In-place version of `scalar_div`.
  fn scalar_div_inplace(&mut self, &N);
}

/**
 * Trait of objects having an addition with a scalar.
 */
pub trait ScalarAdd<N>
{
  /// Gets the result of an addition by a scalar.
  fn scalar_add(&self, &N) -> Self;

  /// In-place version of `scalar_add`.
  fn scalar_add_inplace(&mut self, &N);
}

/**
 * Trait of objects having a subtraction with a scalar.
 */
pub trait ScalarSub<N>
{
  /// Gets the result of a subtraction by a scalar.
  fn scalar_sub(&self, &N) -> Self;

  /// In-place version of `scalar_sub`.
  fn scalar_sub_inplace(&mut self, &N);
}
