/**
 * Trait of objects having a right multiplication with another element.
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 * 
 * ~~~
 * trait Mul<V, M> for M
 * trait Mul<V2, M> for M
 * ~~~
 */
pub trait RMul<V>
{
  /// Computes self * v
  fn rmul(&self, v : &V) -> V;
}

/**
 * Trait of objects having a left multiplication with another element.
 * This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type. The following
 * exemple does not compile (end with an ICE):
 *
 * ~~~
 * trait Mul<V, M> for M
 * trait Mul<V2, M> for M
 * ~~~
 */
pub trait LMul<V>
{
  /// Computes v * self
  fn lmul(&self, &V) -> V;
}
