use super::Permutation;

#[test]
fn test_permutation_4x3() {
    #[rustfmt::skip]
    let mat =
            // col idx     1   2   3 (lapack: 1 based indexing)
        nalgebra::matrix![ 1., 2., 3.;   // 1
                           4., 5., 6.;   // 2
                           7., 8., 9.;   // 3
                           10.,11.,12.]; // 4

    let jpvt = nalgebra::vector![3, 1, 2];
    let mut perm = Permutation::new(jpvt);
    let mut mat2 = mat.clone();
    perm.permute_cols_mut(&mut mat2).unwrap();

    assert_eq!(mat2.column(0), mat.column(2));
    assert_eq!(mat2.column(1), mat.column(0));
    assert_eq!(mat2.column(2), mat.column(1));

    // now inverse permute, which means we expect same as original matrix again
    perm.inv_permute_cols_mut(&mut mat2).unwrap();
    assert_eq!(mat2, mat);

    // now test column permutation
    let jpvt = nalgebra::vector![4, 2, 1, 3];
    let mut perm = Permutation::new(jpvt);

    perm.permute_rows_mut(&mut mat2).unwrap();
    assert_eq!(mat2.row(0), mat.row(3));
    assert_eq!(mat2.row(1), mat.row(1));
    assert_eq!(mat2.row(2), mat.row(0));
    assert_eq!(mat2.row(3), mat.row(2));

    perm.inv_permute_rows_mut(&mut mat2).unwrap();
    assert_eq!(mat2, mat);
}

#[test]
fn test_permutation_dynamic_3x5() {
    use nalgebra::{DMatrix, DVector};

    // Create a 3x5 dynamic matrix with a different pattern
    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 5, &[
        // col idx     1    2    3    4    5  (lapack: 1 based indexing)
                       10., 20., 30., 40., 50.,     // row 1
                       100.,200.,300.,400.,500.,    // row 2
                       1000.,2000.,3000.,4000.,5000.// row 3
    ]);

    // Test column permutation: [5, 2, 1, 4, 3] - rearrange columns
    let jpvt_cols = DVector::from_vec(vec![5, 2, 1, 4, 3]);
    let mut perm = Permutation::new(jpvt_cols);
    let mut mat2 = mat.clone();

    perm.permute_cols_mut(&mut mat2).unwrap();

    assert_eq!(mat2.column(0), mat.column(4));
    assert_eq!(mat2.column(1), mat.column(1));
    assert_eq!(mat2.column(2), mat.column(0));
    assert_eq!(mat2.column(3), mat.column(3));
    assert_eq!(mat2.column(4), mat.column(2));

    perm.inv_permute_cols_mut(&mut mat2).unwrap();
    assert_eq!(mat2, mat);

    let jpvt_rows = DVector::from_vec(vec![2, 3, 1]);
    let mut perm = Permutation::new(jpvt_rows);

    perm.permute_rows_mut(&mut mat2).unwrap();

    assert_eq!(mat2.row(0), mat.row(1));
    assert_eq!(mat2.row(1), mat.row(2));
    assert_eq!(mat2.row(2), mat.row(0));

    // Inverse permute should restore original matrix
    perm.inv_permute_rows_mut(&mut mat2).unwrap();
    assert_eq!(mat2, mat);
}
