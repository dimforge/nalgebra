use lower_triangular::LowerTriangularMat;

pub trait Crout<N> {
    /// Crout LDL* factorization for a symmetric definite indefinite matrix.
    fn crout(self, &mut DiagonalMat<N>) -> LowerTriangularMat<N>;
}
