use core::Matrix;
use core::dimension::{Dynamic, U1, U2, U3, U4, U5, U6};
use core::matrix_vec::MatrixVec;
use core::storage::Owned;
use core::matrix_slice::{SliceStorage, SliceStorageMut};

/*
 *
 *
 * Matrix slice aliases.
 *
 *
 */
/// A column-major matrix slice with `R` rows and `C` columns.
pub type MatrixSliceMN<'a, N, R, C, RStride = U1, CStride = R>
    = Matrix<N, R, C, SliceStorage<'a, N, R, C, RStride, CStride>>;

pub type MatrixSliceN<'a, N, D, RStride = U1, CStride = D> = MatrixSliceMN<'a, N, D, D, RStride, CStride>;

pub type DMatrixSlice<'a, N, RStride = U1, CStride = Dynamic> = MatrixSliceN<'a, N, Dynamic, RStride, CStride>;

pub type MatrixSlice1<'a, N, RStride = U1, CStride = U1>  = MatrixSliceN<'a, N, U1, RStride, CStride>;
pub type MatrixSlice2<'a, N, RStride = U1, CStride = U2>  = MatrixSliceN<'a, N, U2, RStride, CStride>;
pub type MatrixSlice3<'a, N, RStride = U1, CStride = U3>  = MatrixSliceN<'a, N, U3, RStride, CStride>;
pub type MatrixSlice4<'a, N, RStride = U1, CStride = U4>  = MatrixSliceN<'a, N, U4, RStride, CStride>;
pub type MatrixSlice5<'a, N, RStride = U1, CStride = U5>  = MatrixSliceN<'a, N, U5, RStride, CStride>;
pub type MatrixSlice6<'a, N, RStride = U1, CStride = U6>  = MatrixSliceN<'a, N, U6, RStride, CStride>;

pub type MatrixSlice1x2<'a, N, RStride = U1, CStride = U1>  = MatrixSliceMN<'a, N, U2, U1, RStride, CStride>;
pub type MatrixSlice1x3<'a, N, RStride = U1, CStride = U1>  = MatrixSliceMN<'a, N, U3, U1, RStride, CStride>;
pub type MatrixSlice1x4<'a, N, RStride = U1, CStride = U1>  = MatrixSliceMN<'a, N, U4, U1, RStride, CStride>;
pub type MatrixSlice1x5<'a, N, RStride = U1, CStride = U1>  = MatrixSliceMN<'a, N, U5, U1, RStride, CStride>;
pub type MatrixSlice1x6<'a, N, RStride = U1, CStride = U1>  = MatrixSliceMN<'a, N, U6, U1, RStride, CStride>;

pub type MatrixSlice2x2<'a, N, RStride = U1, CStride = U2>  = MatrixSliceMN<'a, N, U2, U2, RStride, CStride>;
pub type MatrixSlice2x3<'a, N, RStride = U1, CStride = U2>  = MatrixSliceMN<'a, N, U3, U2, RStride, CStride>;
pub type MatrixSlice2x4<'a, N, RStride = U1, CStride = U2>  = MatrixSliceMN<'a, N, U4, U2, RStride, CStride>;
pub type MatrixSlice2x5<'a, N, RStride = U1, CStride = U2>  = MatrixSliceMN<'a, N, U5, U2, RStride, CStride>;
pub type MatrixSlice2x6<'a, N, RStride = U1, CStride = U2>  = MatrixSliceMN<'a, N, U6, U2, RStride, CStride>;

pub type MatrixSlice3x2<'a, N, RStride = U1, CStride = U3>  = MatrixSliceMN<'a, N, U2, U3, RStride, CStride>;
pub type MatrixSlice3x3<'a, N, RStride = U1, CStride = U3>  = MatrixSliceMN<'a, N, U3, U3, RStride, CStride>;
pub type MatrixSlice3x4<'a, N, RStride = U1, CStride = U3>  = MatrixSliceMN<'a, N, U4, U3, RStride, CStride>;
pub type MatrixSlice3x5<'a, N, RStride = U1, CStride = U3>  = MatrixSliceMN<'a, N, U5, U3, RStride, CStride>;
pub type MatrixSlice3x6<'a, N, RStride = U1, CStride = U3>  = MatrixSliceMN<'a, N, U6, U3, RStride, CStride>;

pub type MatrixSlice4x2<'a, N, RStride = U1, CStride = U4>  = MatrixSliceMN<'a, N, U2, U4, RStride, CStride>;
pub type MatrixSlice4x3<'a, N, RStride = U1, CStride = U4>  = MatrixSliceMN<'a, N, U3, U4, RStride, CStride>;
pub type MatrixSlice4x4<'a, N, RStride = U1, CStride = U4>  = MatrixSliceMN<'a, N, U4, U4, RStride, CStride>;
pub type MatrixSlice4x5<'a, N, RStride = U1, CStride = U4>  = MatrixSliceMN<'a, N, U5, U4, RStride, CStride>;
pub type MatrixSlice4x6<'a, N, RStride = U1, CStride = U4>  = MatrixSliceMN<'a, N, U6, U4, RStride, CStride>;

pub type MatrixSlice5x2<'a, N, RStride = U1, CStride = U5>  = MatrixSliceMN<'a, N, U2, U5, RStride, CStride>;
pub type MatrixSlice5x3<'a, N, RStride = U1, CStride = U5>  = MatrixSliceMN<'a, N, U3, U5, RStride, CStride>;
pub type MatrixSlice5x4<'a, N, RStride = U1, CStride = U5>  = MatrixSliceMN<'a, N, U4, U5, RStride, CStride>;
pub type MatrixSlice5x5<'a, N, RStride = U1, CStride = U5>  = MatrixSliceMN<'a, N, U5, U5, RStride, CStride>;
pub type MatrixSlice5x6<'a, N, RStride = U1, CStride = U5>  = MatrixSliceMN<'a, N, U6, U5, RStride, CStride>;

pub type MatrixSlice6x2<'a, N, RStride = U1, CStride = U6>  = MatrixSliceMN<'a, N, U2, U6, RStride, CStride>;
pub type MatrixSlice6x3<'a, N, RStride = U1, CStride = U6>  = MatrixSliceMN<'a, N, U3, U6, RStride, CStride>;
pub type MatrixSlice6x4<'a, N, RStride = U1, CStride = U6>  = MatrixSliceMN<'a, N, U4, U6, RStride, CStride>;
pub type MatrixSlice6x5<'a, N, RStride = U1, CStride = U6>  = MatrixSliceMN<'a, N, U5, U6, RStride, CStride>;
pub type MatrixSlice6x6<'a, N, RStride = U1, CStride = U6>  = MatrixSliceMN<'a, N, U6, U6, RStride, CStride>;


/*
 *
 *
 * Same thing, but for mutable slices.
 *
 *
 */
pub type MatrixSliceMutMN<'a, N, R, C, RStride = U1, CStride = R>
    = Matrix<N, R, C, SliceStorageMut<'a, N, R, C, RStride, CStride>>;
