/**
 * A delta-transformation is the generalization of rotation. A delta transform
 * can apply any transformation to the object without translating it.
 * In partilular, 0 is neutral wrt. the delta-transform.
 */
pub trait DeltaTransform<DT>
{
  /// Extracts the delta transformation associated with this transformation.
  fn delta_transform(&self) -> DT;
}

/**
 * Trait of delta-transformations on vectors.
 */
pub trait DeltaTransformVector<V>
{
  /// Applies a delta-transform to a vector.
  fn delta_transform_vector(&self, &V) -> V;
}
