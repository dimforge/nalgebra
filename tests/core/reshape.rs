use na::{
    Const, DMatrix, DMatrixSlice, DMatrixSliceMut, Dyn, Dynamic, Matrix, MatrixSlice,
    MatrixSliceMut, SMatrix, SMatrixSlice, SMatrixSliceMut, VecStorage, U3, U4,
};
use nalgebra_macros::matrix;
use simba::scalar::SupersetOf;

const MATRIX: SMatrix<i32, 4, 3> = matrix![
     1,  2,  3;
     4,  5,  6;
     7,  8,  9;
    10, 11, 12
];

const RESHAPED_MATRIX: SMatrix<i32, 3, 4> = matrix![
    1, 10,  8,  6;
    4,  2, 11,  9;
    7,  5,  3, 12
];

// Helper alias for making it easier to specify dynamically allocated matrices with
// different dimension types (unlike DMatrix)
type GenericDMatrix<T, R, C> = Matrix<T, R, C, VecStorage<T, R, C>>;

#[test]
fn reshape_owned() {
    macro_rules! test_reshape {
        ($in_matrix:ty => $out_matrix:ty, $rows:expr, $cols:expr) => {{
            // This is a pretty weird way to convert, but Matrix implements SubsetOf
            let matrix: $in_matrix = MATRIX.to_subset().unwrap();
            let reshaped: $out_matrix = matrix.reshape_generic($rows, $cols);
            assert_eq!(reshaped, RESHAPED_MATRIX);
        }};
    }

    test_reshape!(SMatrix<_, 4, 3> => SMatrix<_, 3, 4>, U3, U4);
    test_reshape!(GenericDMatrix<_, U4, Dyn> => GenericDMatrix<_, Dyn, Dyn>, Dyn(3), Dyn(4));
    test_reshape!(GenericDMatrix<_, U4, Dyn> => GenericDMatrix<_, U3, Dyn>, U3, Dyn(4));
    test_reshape!(GenericDMatrix<_, U4, Dyn> => GenericDMatrix<_, Dyn, U4>, Dyn(3), U4);
    test_reshape!(DMatrix<_> => DMatrix<_>, Dyn(3), Dyn(4));
}

#[test]
fn reshape_slice() {
    macro_rules! test_reshape {
        ($in_slice:ty => $out_slice:ty, $rows:expr, $cols:expr) => {
            // We test both that types check out by being explicit about types
            // and the actual contents of the matrix
            {
                // By constructing the slice from a mutable reference we can obtain *either*
                // an immutable slice or a mutable slice, which simplifies the testing of both
                // types of mutability
                let mut source_matrix = MATRIX.clone();
                let slice: $in_slice = Matrix::from(&mut source_matrix);
                let reshaped: $out_slice = slice.reshape_generic($rows, $cols);
                assert_eq!(reshaped, RESHAPED_MATRIX);
            }
        };
    }

    // Static "source slice"
    test_reshape!(SMatrixSlice<_, 4, 3> => SMatrixSlice<_, 3, 4>, U3, U4);
    test_reshape!(SMatrixSlice<_, 4, 3> => DMatrixSlice<_>, Dynamic::new(3), Dynamic::new(4));
    test_reshape!(SMatrixSlice<_, 4, 3> => MatrixSlice<_, Const<3>, Dynamic>, U3, Dynamic::new(4));
    test_reshape!(SMatrixSlice<_, 4, 3> => MatrixSlice<_, Dynamic, Const<4>>, Dynamic::new(3), U4);
    test_reshape!(SMatrixSliceMut<_, 4, 3> => SMatrixSliceMut<_, 3, 4>, U3, U4);
    test_reshape!(SMatrixSliceMut<_, 4, 3> => DMatrixSliceMut<_>, Dynamic::new(3), Dynamic::new(4));
    test_reshape!(SMatrixSliceMut<_, 4, 3> => MatrixSliceMut<_, Const<3>, Dynamic>, U3, Dynamic::new(4));
    test_reshape!(SMatrixSliceMut<_, 4, 3> => MatrixSliceMut<_, Dynamic, Const<4>>, Dynamic::new(3), U4);

    // Dynamic "source slice"
    test_reshape!(DMatrixSlice<_> => SMatrixSlice<_, 3, 4>, U3, U4);
    test_reshape!(DMatrixSlice<_> => DMatrixSlice<_>, Dynamic::new(3), Dynamic::new(4));
    test_reshape!(DMatrixSlice<_> => MatrixSlice<_, Const<3>, Dynamic>, U3, Dynamic::new(4));
    test_reshape!(DMatrixSlice<_> => MatrixSlice<_, Dynamic, Const<4>>, Dynamic::new(3), U4);
    test_reshape!(DMatrixSliceMut<_> => SMatrixSliceMut<_, 3, 4>, U3, U4);
    test_reshape!(DMatrixSliceMut<_> => DMatrixSliceMut<_>, Dynamic::new(3), Dynamic::new(4));
    test_reshape!(DMatrixSliceMut<_> => MatrixSliceMut<_, Const<3>, Dynamic>, U3, Dynamic::new(4));
    test_reshape!(DMatrixSliceMut<_> => MatrixSliceMut<_, Dynamic, Const<4>>, Dynamic::new(3), U4);
}
