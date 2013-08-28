// FIXME: return an iterator instead
/// Traits of objecs which can form a basis.
pub trait Basis {
    /// Iterate through the canonical basis of the space in which this object lives.
    fn canonical_basis(&fn(Self) -> bool);

    /// Iterate through a basis of the subspace orthogonal to `self`.
    fn orthonormal_subspace_basis(&self, &fn(Self) -> bool);

    /// Creates the canonical basis of the space in which this object lives.
    fn canonical_basis_list() -> ~[Self] {
        let mut res = ~[];

        do Basis::canonical_basis |elem| {
            res.push(elem);

            true
        }

        res
    }

    /// Creates a basis of the subspace orthogonal to `self`.
    fn orthonormal_subspace_basis_list(&self) -> ~[Self] {
        let mut res = ~[];

        do self.orthonormal_subspace_basis |elem| {
            res.push(elem);

            true
        }

        res
    }
}
