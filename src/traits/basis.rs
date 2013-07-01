pub trait Basis
{
  /// Computes the canonical basis of the space in which this object lives.
  // FIXME: implement the for loop protocol?
  fn canonical_basis(&fn(Self));
  fn orthonormal_subspace_basis(&self, &fn(Self));
}
