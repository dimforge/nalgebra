// FIXME: this trait should not be on nalgebra.
// however, it is needed because std::ops::Index is (strangely) to poor: it
// does not have a function to set values.
// Also, using Index with tuples crashes.
pub trait Indexable<Index, Res>
{
  fn at(&self, Index) -> Res;
  fn set(&mut self, Index, Res);
  fn swap(&mut self, Index, Index);
}
