pub trait Transformation<M>
{
  fn transformation(&self) -> M;

  // XXX: we must use "transform_by" instead of "transform" because of a
  // conflict with some iterator function…
  fn transform_by(&mut self, &M);
}

pub trait Transformable<M, Res: Transformation<M>>
{
  fn transformed(&self, &M) -> Res;
}
