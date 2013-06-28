pub trait Transformation<M>
{
  fn transformation(&self) -> M;

  fn inv_transformation(&self) -> M;

  fn transform_by(&mut self, &M);
}

pub trait Transform<V>
{
  // XXX: sadly we cannot call this `transform` as it conflicts with the
  // iterators' `transform` function (which seems always exist).
  fn transform_vec(&self, &V) -> V;
  fn inv_transform(&self, &V) -> V;
}

pub trait Transformable<M, Res: Transformation<M>>
{
  fn transformed(&self, &M) -> Res;
}
