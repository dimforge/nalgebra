pub trait Basis
{
  /// Computes the canonical basis of the space in which this object lives.
  // FIXME: need type-associated values
  // FIXME: this will make allocations… this is bad
  fn canonical_basis()                -> ~[Self];
  fn orthogonal_subspace_basis(&self) -> ~[Self];
}
