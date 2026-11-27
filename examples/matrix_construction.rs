extern crate nalgebra as na;

use na::{DMatrix, DVector, Matrix2x3, RowVector3, Vector2};

fn main() {
    // All the following matrices are equal but constructed in different ways.
    // For small matrices known directly from their entries, the matrix! macro
    // is the preferred construction method. Entries are written row-by-row,
    // with commas separating columns and semicolons separating rows.
    let m = na::matrix![1.1, 1.2, 1.3;
                        2.1, 2.2, 2.3];

    let m1 = Matrix2x3::new(1.1, 1.2, 1.3, 2.1, 2.2, 2.3);

    let m2 = Matrix2x3::from_rows(&[
        RowVector3::new(1.1, 1.2, 1.3),
        RowVector3::new(2.1, 2.2, 2.3),
    ]);

    let m3 = Matrix2x3::from_columns(&[
        Vector2::new(1.1, 2.1),
        Vector2::new(1.2, 2.2),
        Vector2::new(1.3, 2.3),
    ]);

    let m4 = Matrix2x3::from_row_slice(&[1.1, 1.2, 1.3, 2.1, 2.2, 2.3]);

    let m5 = Matrix2x3::from_column_slice(&[1.1, 2.1, 1.2, 2.2, 1.3, 2.3]);

    let m6 = Matrix2x3::from_fn(|r, c| (r + 1) as f32 + (c + 1) as f32 / 10.0);

    let m7 = Matrix2x3::from_iterator([1.1f32, 2.1, 1.2, 2.2, 1.3, 2.3].iter().cloned());

    assert_eq!(m, m1);
    assert_eq!(m, m2);
    assert_eq!(m, m3);
    assert_eq!(m, m4);
    assert_eq!(m, m5);
    assert_eq!(m, m6);
    assert_eq!(m, m7);

    // For fixed-size vectors, use vector!. Dynamic vectors use dvector!.
    let v = na::vector![1.0, 2.0];
    let dv = na::dvector![1.0, 2.0, 3.0];

    assert_eq!(v, Vector2::new(1.0, 2.0));
    assert_eq!(dv, DVector::from_column_slice(&[1.0, 2.0, 3.0]));

    // All the following matrices are equal but constructed in different ways.
    // This time, we used a dynamically-sized matrix to show the extra arguments
    // for the matrix shape.
    let dm = na::dmatrix![1.0, 0.0, 0.0;
                          0.0, 1.0, 0.0;
                          0.0, 0.0, 1.0;
                          0.0, 0.0, 0.0];

    let dm1 = DMatrix::from_row_slice(
        4,
        3,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    );
    let dm2 = DMatrix::from_diagonal_element(4, 3, 1.0);
    let dm3 = DMatrix::identity(4, 3);
    let dm4 = DMatrix::from_fn(4, 3, |r, c| if r == c { 1.0 } else { 0.0 });
    let dm5 = DMatrix::from_iterator(
        4,
        3,
        [
            // Components listed column-by-column.
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]
        .iter()
        .cloned(),
    );

    assert_eq!(dm, dm1);
    assert_eq!(dm, dm2);
    assert_eq!(dm, dm3);
    assert_eq!(dm, dm4);
    assert_eq!(dm, dm5);
}
