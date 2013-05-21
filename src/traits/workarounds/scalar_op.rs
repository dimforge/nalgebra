/**
 * Trait of objects having a multiplication with a scalar.
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Mul<V, T> for T
 * trait Mul<V2, T> for T
 * ~~~
 */
pub trait ScalarMul<T>
{
  /// Gets the result of a multiplication by a scalar.
  fn scalar_mul(&self, &T) -> Self;

  /// In-place version of `scalar_mul`.
  fn scalar_mul_inplace(&mut self, &T);
}

/**
 * Trait of objects having a division with a scalar.
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Div<V, T> for T
 * trait Div<V2, T> for T
 * ~~~
 */
pub trait ScalarDiv<T>
{
  /// Gets the result of a division by a scalar.
  fn scalar_div(&self, &T) -> Self;

  /// In-place version of `scalar_div`.
  fn scalar_div_inplace(&mut self, &T);
}

/**
 * Trait of objects having an addition with a scalar.
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Add<V, T> for T
 * trait Add<V2, T> for T
 * ~~~
 */
pub trait ScalarAdd<T>
{
  /// Gets the result of an addition by a scalar.
  fn scalar_add(&self, &T) -> Self;

  /// In-place version of `scalar_add`.
  fn scalar_add_inplace(&mut self, &T);
}

/**
 * Trait of objects having a subtraction with a scalar.
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Sub<V, T> for T
 * trait Sub<V2, T> for T
 * ~~~
 */
pub trait ScalarSub<T>
{
  /// Gets the result of a subtraction by a scalar.
  fn scalar_sub(&self, &T) -> Self;

  /// In-place version of `scalar_sub`.
  fn scalar_sub_inplace(&mut self, &T);
}
