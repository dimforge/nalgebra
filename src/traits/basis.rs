pub trait Basis
{
  fn canonical_basis()                -> ~[Self]; // FIXME: is it the right pointer?
  fn orthogonal_subspace_basis(&self) -> ~[Self];
}
