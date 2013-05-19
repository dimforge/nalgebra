/// A delta-transformation is the generalization of rotation. A delta transform
/// can apply any transformation to the object without translating it.
/// In partilular, 0 is neutral wrt. the delta-transform.
pub trait DeltaTransform<DT>
{
  /// Extracts the delta transformation associated with this transformation.
  fn delta_transform(&self) -> DT;
  // FIXME: add functions to apply the delta-transform to a vector without
  // explicit computation of the transform (does this avoid some matrix copy?)
}
