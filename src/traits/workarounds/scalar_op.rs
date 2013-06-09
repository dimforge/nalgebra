/**
 * Trait of objects having a multiplication with a scalar.
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Mul<V, N> for N
 * trait Mul<V2, N> for N
 * ~~~
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
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Div<V, N> for N
 * trait Div<V2, N> for N
 * ~~~
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
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Add<V, N> for N
 * trait Add<V2, N> for N
 * ~~~
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
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Sub<V, N> for N
 * trait Sub<V2, N> for N
 * ~~~
 */
pub trait ScalarSub<N>
{
  /// Gets the result of a subtraction by a scalar.
  fn scalar_sub(&self, &N) -> Self;

  /// In-place version of `scalar_sub`.
  fn scalar_sub_inplace(&mut self, &N);
}
