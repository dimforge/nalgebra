/** This is a workaround of the fact we cannot implement the same trait
 * (with different type parameters) twice for the same type:
 * ~~~
 * trait Mul<V, V> for M
 * trait Mul<V2, V2> for M
 * ~~~
 */
pub trait RMul<V>
{
  /// Computes self * v
  fn rmul(&self, v : &V) -> V;
}

pub trait LMul<V>
{
  /// Computes v * self
  fn lmul(&self, &V) -> V;
}
