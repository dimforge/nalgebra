use encase::{
    matrix::{AsMutMatrixParts, AsRefMatrixParts, FromMatrixParts, MatrixScalar, impl_matrix},
    vector::{AsMutVectorParts, AsRefVectorParts, FromVectorParts, VectorScalar, impl_vector},
};

impl_vector!(2, crate::VectorView2<'_, T>);
impl_vector!(2, crate::VectorViewMut2<'_, T>);
impl_vector!(2, crate::Vector2<T>);
impl_vector!(2, crate::Point2<T>; (T: crate::Scalar));

impl_vector!(3, crate::VectorView3<'_, T>);
impl_vector!(3, crate::VectorViewMut3<'_, T>);
impl_vector!(3, crate::Vector3<T>);
impl_vector!(3, crate::Point3<T>; (T: crate::Scalar));

impl_vector!(4, crate::VectorView4<'_, T>);
impl_vector!(4, crate::VectorViewMut4<'_, T>);
impl_vector!(4, crate::Vector4<T>);
impl_vector!(4, crate::Point4<T>; (T: crate::Scalar));

impl_matrix!(2, 2, crate::MatrixView2<'_, T>);
impl_matrix!(2, 2, crate::MatrixViewMut2<'_, T>);
impl_matrix!(2, 2, crate::Matrix2<T>);

impl_matrix!(3, 2, crate::MatrixView2x3<'_, T>);
impl_matrix!(4, 2, crate::MatrixView2x4<'_, T>);
impl_matrix!(2, 3, crate::MatrixView3x2<'_, T>);
impl_matrix!(3, 2, crate::MatrixViewMut2x3<'_, T>);
impl_matrix!(4, 2, crate::MatrixViewMut2x4<'_, T>);
impl_matrix!(2, 3, crate::MatrixViewMut3x2<'_, T>);
impl_matrix!(3, 2, crate::Matrix2x3<T>);
impl_matrix!(4, 2, crate::Matrix2x4<T>);
impl_matrix!(2, 3, crate::Matrix3x2<T>);

impl_matrix!(3, 3, crate::MatrixView3<'_, T>);
impl_matrix!(3, 3, crate::MatrixViewMut3<'_, T>);
impl_matrix!(3, 3, crate::Matrix3<T>);

impl_matrix!(4, 3, crate::MatrixView3x4<'_, T>);
impl_matrix!(2, 4, crate::MatrixView4x2<'_, T>);
impl_matrix!(3, 4, crate::MatrixView4x3<'_, T>);
impl_matrix!(4, 3, crate::MatrixViewMut3x4<'_, T>);
impl_matrix!(2, 4, crate::MatrixViewMut4x2<'_, T>);
impl_matrix!(3, 4, crate::MatrixViewMut4x3<'_, T>);
impl_matrix!(4, 3, crate::Matrix3x4<T>);
impl_matrix!(2, 4, crate::Matrix4x2<T>);
impl_matrix!(3, 4, crate::Matrix4x3<T>);

impl_matrix!(4, 4, crate::MatrixView4<'_, T>);
impl_matrix!(4, 4, crate::MatrixViewMut4<'_, T>);
impl_matrix!(4, 4, crate::Matrix4<T>);

impl<T: VectorScalar + crate::Scalar, const N: usize> FromVectorParts<T, N> for crate::Point<T, N>
where
    crate::SVector<T, N>: FromVectorParts<T, N>,
{
    fn from_parts(parts: [T; N]) -> Self {
        <crate::SVector<T, N> as FromVectorParts<T, N>>::from_parts(parts).into()
    }
}

impl<T: VectorScalar + crate::Scalar, const N: usize> AsRefVectorParts<T, N> for crate::Point<T, N>
where
    crate::SVector<T, N>: AsRefVectorParts<T, N>,
{
    fn as_ref_parts(&self) -> &[T; N] {
        self.coords.as_ref_parts()
    }
}

impl<T: VectorScalar + crate::Scalar, const N: usize> AsMutVectorParts<T, N> for crate::Point<T, N>
where
    crate::SVector<T, N>: AsMutVectorParts<T, N>,
{
    fn as_mut_parts(&mut self) -> &mut [T; N] {
        self.coords.as_mut_parts()
    }
}

impl<T: VectorScalar, S, const N: usize> AsRefVectorParts<T, N>
    for crate::Matrix<T, crate::Const<N>, crate::Const<1>, S>
where
    Self: AsRef<[T; N]>,
{
    fn as_ref_parts(&self) -> &[T; N] {
        self.as_ref()
    }
}

impl<T: VectorScalar, S, const N: usize> AsMutVectorParts<T, N>
    for crate::Matrix<T, crate::Const<N>, crate::Const<1>, S>
where
    Self: AsMut<[T; N]>,
{
    fn as_mut_parts(&mut self) -> &mut [T; N] {
        self.as_mut()
    }
}

impl<T: VectorScalar, const N: usize> FromVectorParts<T, N> for crate::SMatrix<T, N, 1> {
    fn from_parts(parts: [T; N]) -> Self {
        Self::from_array_storage(crate::ArrayStorage([parts]))
    }
}

impl<T: MatrixScalar, S, const C: usize, const R: usize> AsRefMatrixParts<T, C, R>
    for crate::Matrix<T, crate::Const<R>, crate::Const<C>, S>
where
    Self: AsRef<[[T; R]; C]>,
{
    fn as_ref_parts(&self) -> &[[T; R]; C] {
        self.as_ref()
    }
}

impl<T: MatrixScalar, S, const C: usize, const R: usize> AsMutMatrixParts<T, C, R>
    for crate::Matrix<T, crate::Const<R>, crate::Const<C>, S>
where
    Self: AsMut<[[T; R]; C]>,
{
    fn as_mut_parts(&mut self) -> &mut [[T; R]; C] {
        self.as_mut()
    }
}

impl<T: MatrixScalar, const C: usize, const R: usize> FromMatrixParts<T, C, R>
    for crate::SMatrix<T, R, C>
{
    fn from_parts(parts: [[T; R]; C]) -> Self {
        Self::from_array_storage(crate::ArrayStorage(parts))
    }
}
