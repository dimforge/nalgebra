pub trait Basis
{
  /// Computes the canonical basis of the space in which this object lives.
  // FIXME: need type-associated values
  fn canonical_basis()                -> ~[Self];
  fn orthogonal_subspace_basis(&self) -> ~[Self];
}
