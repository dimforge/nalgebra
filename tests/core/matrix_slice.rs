use na::{U2, U3, U4};
use na::{DMatrix,
         RowVector4,
         Vector3,
         Matrix2, Matrix3,
         Matrix3x4, Matrix4x2, Matrix2x4, Matrix6x2, Matrix2x6};

#[test]
fn nested_fixed_slices() {
    let a = Matrix3x4::new(11.0, 12.0, 13.0, 14.0,
                           21.0, 22.0, 23.0, 24.0,
                           31.0, 32.0, 33.0, 34.0);

    let s1 = a.fixed_slice::<U3, U3>(0, 1);                       // Simple slice.
    let s2 = s1.fixed_slice::<U2, U2>(1, 1);                      // Slice of slice.
    let s3 = s1.fixed_slice_with_steps::<U2, U2>((0, 0), (1, 1)); // Slice of slice with steps.

    let expected_owned_s1 = Matrix3::new(12.0, 13.0, 14.0,
                                         22.0, 23.0, 24.0,
                                         32.0, 33.0, 34.0);

    let expected_owned_s2 = Matrix2::new(23.0, 24.0,
                                         33.0, 34.0);

    let expected_owned_s3 = Matrix2::new(12.0, 14.0,
                                         32.0, 34.0);

    assert_eq!(expected_owned_s1, s1.clone_owned());
    assert_eq!(expected_owned_s2, s2.clone_owned());
    assert_eq!(expected_owned_s3, s3.clone_owned());
}

#[test]
fn nested_slices() {
    let a = Matrix3x4::new(11.0, 12.0, 13.0, 14.0,
                           21.0, 22.0, 23.0, 24.0,
                           31.0, 32.0, 33.0, 34.0);

    let s1 = a.slice((0, 1), (3, 3));
    let s2 = s1.slice((1, 1), (2, 2));
    let s3 = s1.slice_with_steps((0, 0), (2, 2), (1, 1));

    let expected_owned_s1 = DMatrix::from_row_slice(3, 3, &[ 12.0, 13.0, 14.0,
                                                             22.0, 23.0, 24.0,
                                                             32.0, 33.0, 34.0 ]);

    let expected_owned_s2 = DMatrix::from_row_slice(2, 2, &[ 23.0, 24.0,
                                                             33.0, 34.0 ]);

    let expected_owned_s3 = DMatrix::from_row_slice(2, 2, &[ 12.0, 14.0,
                                                             32.0, 34.0 ]);

    assert_eq!(expected_owned_s1, s1.clone_owned());
    assert_eq!(expected_owned_s2, s2.clone_owned());
    assert_eq!(expected_owned_s3, s3.clone_owned());
}

#[test]
fn slice_mut() {
    let mut a = Matrix3x4::new(11.0, 12.0, 13.0, 14.0,
                               21.0, 22.0, 23.0, 24.0,
                               31.0, 32.0, 33.0, 34.0);

    {
        // We modify `a` through the mutable slice.
        let mut s1 = a.slice_with_steps_mut((0, 1), (2, 2), (1, 1));
        s1.fill(0.0);
    }

    let expected_a = Matrix3x4::new(11.0,  0.0, 13.0,  0.0,
                                    21.0, 22.0, 23.0, 24.0,
                                    31.0,  0.0, 33.0,  0.0);

    assert_eq!(expected_a, a);
}

#[test]
fn nested_row_slices() {
    let a = Matrix6x2::new(11.0, 12.0,
                           21.0, 22.0,
                           31.0, 32.0,
                           41.0, 42.0,
                           51.0, 52.0,
                           61.0, 62.0);
    let s1 = a.fixed_rows::<U4>(1);
    let s2 = s1.fixed_rows_with_step::<U2>(1, 1);

    let expected_owned_s1 = Matrix4x2::new(21.0, 22.0,
                                           31.0, 32.0,
                                           41.0, 42.0,
                                           51.0, 52.0);

    let expected_owned_s2 = Matrix2::new(31.0, 32.0,
                                         51.0, 52.0);

    assert_eq!(expected_owned_s1, s1.clone_owned());
    assert_eq!(expected_owned_s2, s2.clone_owned());
}

#[test]
fn row_slice_mut() {
    let mut a = Matrix6x2::new(11.0, 12.0,
                               21.0, 22.0,
                               31.0, 32.0,
                               41.0, 42.0,
                               51.0, 52.0,
                               61.0, 62.0);
    {
        // We modify `a` through the mutable slice.
        let mut s1 = a.rows_with_step_mut(1, 3, 1);
        s1.fill(0.0);
    }

    let expected_a = Matrix6x2::new(11.0, 12.0,
                                     0.0,  0.0,
                                    31.0, 32.0,
                                     0.0,  0.0,
                                    51.0, 52.0,
                                     0.0,  0.0);

    assert_eq!(expected_a, a);
}

#[test]
fn nested_col_slices() {
    let a = Matrix2x6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                           21.0, 22.0, 23.0, 24.0, 25.0, 26.0);
    let s1 = a.fixed_columns::<U4>(1);
    let s2 = s1.fixed_columns_with_step::<U2>(1, 1);

    let expected_owned_s1 = Matrix2x4::new(12.0, 13.0, 14.0, 15.0,
                                           22.0, 23.0, 24.0, 25.0);

    let expected_owned_s2 = Matrix2::new(13.0, 15.0,
                                         23.0, 25.0);

    assert_eq!(expected_owned_s1, s1.clone_owned());
    assert_eq!(expected_owned_s2, s2.clone_owned());
}

#[test]
fn col_slice_mut() {
    let mut a = Matrix2x6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                               21.0, 22.0, 23.0, 24.0, 25.0, 26.0);

    {
        // We modify `a` through the mutable slice.
        let mut s1 = a.columns_with_step_mut(1, 3, 1);
        s1.fill(0.0);
    }

    let expected_a = Matrix2x6::new(11.0, 0.0, 13.0, 0.0, 15.0, 0.0,
                                    21.0, 0.0, 23.0, 0.0, 25.0, 0.0);

    assert_eq!(expected_a, a.clone_owned());
}

#[test]
fn rows_range_pair() {
    let a = Matrix3x4::new(11.0, 12.0, 13.0, 14.0,
                           21.0, 22.0, 23.0, 24.0,
                           31.0, 32.0, 33.0, 34.0);

    let (l, r) = a.rows_range_pair(.. 3, 3 ..);
    assert!(r.len() == 0 && l.eq(&a));

    let (l, r) = a.rows_range_pair(0, 1 ..);

    let expected_l = RowVector4::new(11.0, 12.0, 13.0, 14.0);
    let expected_r = Matrix2x4::new(21.0, 22.0, 23.0, 24.0,
                                    31.0, 32.0, 33.0, 34.0);
    assert!(l.eq(&expected_l) && r.eq(&expected_r));
}

#[test]
fn columns_range_pair() {
    let a = Matrix3x4::new(11.0, 12.0, 13.0, 14.0,
                           21.0, 22.0, 23.0, 24.0,
                           31.0, 32.0, 33.0, 34.0);

    let (l, r) = a.columns_range_pair(.. 4, 4 ..);
    assert!(r.len() == 0 && l.eq(&a));

    let (l, r) = a.columns_range_pair(0, 1 ..);

    let expected_l = Vector3::new(11.0, 21.0, 31.0);
    let expected_r = Matrix3::new(12.0, 13.0, 14.0,
                                  22.0, 23.0, 24.0,
                                  32.0, 33.0, 34.0);
    assert!(l.eq(&expected_l) && r.eq(&expected_r));
}

#[test]
#[should_panic]
fn row_out_of_bounds() {
    let a = Matrix3x4::<f32>::zeros();
    a.row(3);
}

#[test]
#[should_panic]
fn rows_out_of_bounds() {
    let a = Matrix3x4::<f32>::zeros();
    a.rows(1, 3);
}

#[test]
#[should_panic]
fn rows_with_step_out_of_bounds() {
    let a = Matrix3x4::<f32>::zeros();
    a.rows_with_step(1, 2, 1);
}

#[test]
#[should_panic]
fn column_out_of_bounds() {
    let a = Matrix3x4::<f32>::zeros();
    a.column(4);
}

#[test]
#[should_panic]
fn columns_out_of_bounds() {
    let a = Matrix3x4::<f32>::zeros();
    a.columns(2, 3);
}

#[test]
#[should_panic]
fn columns_with_step_out_of_bounds() {
    let a = Matrix3x4::<f32>::zeros();
    a.columns_with_step(2, 2, 1);
}

#[test]
#[should_panic]
fn slice_out_of_bounds() {
    let a = Matrix3x4::<f32>::zeros();
    a.slice((1, 2), (3, 1));
}

#[test]
#[should_panic]
fn slice_with_steps_out_of_bounds() {
    let a = Matrix3x4::<f32>::zeros();
    a.slice_with_steps((1, 2), (2, 2), (0, 1));
}
