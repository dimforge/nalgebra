use na::{Matrix,
         DMatrix,
         Matrix3, Matrix4, Matrix5,
         Matrix4x3, Matrix3x4, Matrix5x3, Matrix3x5, Matrix4x5, Matrix5x4};
use na::{Dynamic, U2, U3, U5};

#[test]
fn upper_lower_triangular() {
    let m = Matrix4::new(
        11.0, 12.0, 13.0, 14.0,
        21.0, 22.0, 23.0, 24.0,
        31.0, 32.0, 33.0, 34.0,
        41.0, 42.0, 43.0, 44.0);

    let um = Matrix4::new(
        11.0, 12.0, 13.0, 14.0,
         0.0, 22.0, 23.0, 24.0,
         0.0,  0.0, 33.0, 34.0,
         0.0,  0.0,  0.0, 44.0);

    let lm = Matrix4::new(
        11.0,  0.0,  0.0,  0.0,
        21.0, 22.0,  0.0,  0.0,
        31.0, 32.0, 33.0,  0.0,
        41.0, 42.0, 43.0, 44.0);

    let computed_um = m.upper_triangle();
    let computed_lm = m.lower_triangle();

    assert_eq!(um, computed_um);
    assert_eq!(lm, computed_lm);

    let symm_um = Matrix4::new(
        11.0, 12.0, 13.0, 14.0,
        12.0, 22.0, 23.0, 24.0,
        13.0, 23.0, 33.0, 34.0,
        14.0, 24.0, 34.0, 44.0);

    let symm_lm = Matrix4::new(
        11.0, 21.0, 31.0, 41.0,
        21.0, 22.0, 32.0, 42.0,
        31.0, 32.0, 33.0, 43.0,
        41.0, 42.0, 43.0, 44.0);

    let mut computed_symm_um = m.clone();
    let mut computed_symm_lm = m.clone();

    computed_symm_um.fill_lower_triangle_with_upper_triangle();
    computed_symm_lm.fill_upper_triangle_with_lower_triangle();
    assert_eq!(symm_um, computed_symm_um);
    assert_eq!(symm_lm, computed_symm_lm);


    let m = Matrix5x3::new(
        11.0, 12.0, 13.0,
        21.0, 22.0, 23.0,
        31.0, 32.0, 33.0,
        41.0, 42.0, 43.0,
        51.0, 52.0, 53.0);

    let um = Matrix5x3::new(
        11.0, 12.0, 13.0,
         0.0, 22.0, 23.0,
         0.0,  0.0, 33.0,
         0.0,  0.0,  0.0,
         0.0,  0.0,  0.0);

    let lm = Matrix5x3::new(
        11.0,  0.0,  0.0,
        21.0, 22.0,  0.0,
        31.0, 32.0, 33.0,
        41.0, 42.0, 43.0,
        51.0, 52.0, 53.0);

    let computed_um = m.upper_triangle();
    let computed_lm = m.lower_triangle();

    assert_eq!(um, computed_um);
    assert_eq!(lm, computed_lm);


    let m = Matrix3x5::new(
        11.0, 12.0, 13.0, 14.0, 15.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
        31.0, 32.0, 33.0, 34.0, 35.0);

    let um = Matrix3x5::new(
        11.0, 12.0, 13.0, 14.0, 15.0,
         0.0, 22.0, 23.0, 24.0, 25.0,
         0.0,  0.0, 33.0, 34.0, 35.0);

    let lm = Matrix3x5::new(
        11.0,  0.0,  0.0,  0.0, 0.0,
        21.0, 22.0,  0.0,  0.0, 0.0,
        31.0, 32.0, 33.0,  0.0, 0.0);

    let computed_um = m.upper_triangle();
    let computed_lm = m.lower_triangle();

    assert_eq!(um, computed_um);
    assert_eq!(lm, computed_lm);

    let mut m = Matrix4x5::new(
        11.0, 12.0, 13.0, 14.0, 15.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
        31.0, 32.0, 33.0, 34.0, 35.0,
        41.0, 42.0, 43.0, 44.0, 45.0);

    let expected_m = Matrix4x5::new(
        11.0, 12.0,  0.0,  0.0,  0.0,
        21.0, 22.0, 23.0,  0.0,  0.0,
        31.0, 32.0, 33.0, 34.0,  0.0,
        41.0, 42.0, 43.0, 44.0, 45.0);

    m.fill_upper_triangle(0.0, 2);

    assert_eq!(m, expected_m);

    let mut m = Matrix4x5::new(
        11.0, 12.0, 13.0, 14.0, 15.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
        31.0, 32.0, 33.0, 34.0, 35.0,
        41.0, 42.0, 43.0, 44.0, 45.0);

    let expected_m = Matrix4x5::new(
        11.0, 12.0, 13.0, 14.0, 15.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
         0.0, 32.0, 33.0, 34.0, 35.0,
         0.0,  0.0, 43.0, 44.0, 45.0);

    m.fill_lower_triangle(0.0, 2);

    assert_eq!(m, expected_m);

    let mut m = Matrix5x4::new(
        11.0, 12.0, 13.0, 14.0,
        21.0, 22.0, 23.0, 24.0,
        31.0, 32.0, 33.0, 34.0,
        41.0, 42.0, 43.0, 44.0,
        51.0, 52.0, 53.0, 54.0);

    let expected_m = Matrix5x4::new(
        11.0, 12.0,  0.0,  0.0,
        21.0, 22.0, 23.0,  0.0,
        31.0, 32.0, 33.0, 34.0,
        41.0, 42.0, 43.0, 44.0,
        51.0, 52.0, 53.0, 54.0);

    m.fill_upper_triangle(0.0, 2);

    assert_eq!(m, expected_m);

    let mut m = Matrix5x4::new(
        11.0, 12.0, 13.0, 14.0,
        21.0, 22.0, 23.0, 24.0,
        31.0, 32.0, 33.0, 34.0,
        41.0, 42.0, 43.0, 44.0,
        51.0, 52.0, 53.0, 54.0);

    let expected_m = Matrix5x4::new(
        11.0, 12.0, 13.0, 14.0,
        21.0, 22.0, 23.0, 24.0,
         0.0, 32.0, 33.0, 34.0,
         0.0,  0.0, 43.0, 44.0,
         0.0,  0.0,  0.0, 54.0);

    m.fill_lower_triangle(0.0, 2);

    assert_eq!(m, expected_m);
}

#[test]
fn swap_rows() {
    let mut m = Matrix5x3::new(
        11.0, 12.0, 13.0,
        21.0, 22.0, 23.0,
        31.0, 32.0, 33.0,
        41.0, 42.0, 43.0,
        51.0, 52.0, 53.0);

    let expected = Matrix5x3::new(
        11.0, 12.0, 13.0,
        41.0, 42.0, 43.0,
        31.0, 32.0, 33.0,
        21.0, 22.0, 23.0,
        51.0, 52.0, 53.0);

    m.swap_rows(1, 3);

    assert_eq!(m, expected);
}

#[test]
fn swap_columns() {
    let mut m = Matrix3x5::new(
        11.0, 12.0, 13.0, 14.0, 15.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
        31.0, 32.0, 33.0, 34.0, 35.0);

    let expected = Matrix3x5::new(
        11.0, 14.0, 13.0, 12.0, 15.0,
        21.0, 24.0, 23.0, 22.0, 25.0,
        31.0, 34.0, 33.0, 32.0, 35.0);

    m.swap_columns(1, 3);

    assert_eq!(m, expected);
}

#[test]
fn remove_columns() {
    let m = Matrix3x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let expected1 = Matrix3x4::new(
        12, 13, 14, 15,
        22, 23, 24, 25,
        32, 33, 34, 35);

    let expected2 = Matrix3x4::new(
        11, 12, 13, 14,
        21, 22, 23, 24,
        31, 32, 33, 34);

    let expected3 = Matrix3x4::new(
        11, 12, 14, 15,
        21, 22, 24, 25,
        31, 32, 34, 35);

    assert_eq!(m.remove_column(0), expected1);
    assert_eq!(m.remove_column(4), expected2);
    assert_eq!(m.remove_column(2), expected3);

    let expected1 = Matrix3::new(
        13, 14, 15,
        23, 24, 25,
        33, 34, 35);

    let expected2 = Matrix3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33);

    let expected3 = Matrix3::new(
        11, 12, 15,
        21, 22, 25,
        31, 32, 35);

    assert_eq!(m.remove_fixed_columns::<U2>(0), expected1);
    assert_eq!(m.remove_fixed_columns::<U2>(3), expected2);
    assert_eq!(m.remove_fixed_columns::<U2>(2), expected3);

    // The following is just to verify that the return type dimensions is correctly inferred.
    let computed: Matrix<_, U3, Dynamic, _> = m.remove_columns(3, 2);
    assert!(computed.eq(&expected2));
}


#[test]
fn remove_rows() {
    let m = Matrix5x3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        51, 52, 53);

    let expected1 = Matrix4x3::new(
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        51, 52, 53);

    let expected2 = Matrix4x3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43);

    let expected3 = Matrix4x3::new(
        11, 12, 13,
        21, 22, 23,
        41, 42, 43,
        51, 52, 53);

    assert_eq!(m.remove_row(0), expected1);
    assert_eq!(m.remove_row(4), expected2);
    assert_eq!(m.remove_row(2), expected3);

    let expected1 = Matrix3::new(
        31, 32, 33,
        41, 42, 43,
        51, 52, 53);

    let expected2 = Matrix3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33);

    let expected3 = Matrix3::new(
        11, 12, 13,
        21, 22, 23,
        51, 52, 53);

    assert_eq!(m.remove_fixed_rows::<U2>(0), expected1);
    assert_eq!(m.remove_fixed_rows::<U2>(3), expected2);
    assert_eq!(m.remove_fixed_rows::<U2>(2), expected3);

    // The following is just to verify that the return type dimensions is correctly inferred.
    let computed: Matrix<_, Dynamic, U3, _> = m.remove_rows(3, 2);
    assert!(computed.eq(&expected2));
}


#[test]
fn insert_columns() {
    let m = Matrix5x3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        51, 52, 53);

    let expected1 = Matrix5x4::new(
        0, 11, 12, 13,
        0, 21, 22, 23,
        0, 31, 32, 33,
        0, 41, 42, 43,
        0, 51, 52, 53);

    let expected2 = Matrix5x4::new(
        11, 12, 13, 0,
        21, 22, 23, 0,
        31, 32, 33, 0,
        41, 42, 43, 0,
        51, 52, 53, 0);

    let expected3 = Matrix5x4::new(
        11, 12, 0, 13,
        21, 22, 0, 23,
        31, 32, 0, 33,
        41, 42, 0, 43,
        51, 52, 0, 53);

    assert_eq!(m.insert_column(0, 0), expected1);
    assert_eq!(m.insert_column(3, 0), expected2);
    assert_eq!(m.insert_column(2, 0), expected3);

    let expected1 = Matrix5::new(
        0, 0, 11, 12, 13,
        0, 0, 21, 22, 23,
        0, 0, 31, 32, 33,
        0, 0, 41, 42, 43,
        0, 0, 51, 52, 53);

    let expected2 = Matrix5::new(
        11, 12, 13, 0, 0,
        21, 22, 23, 0, 0,
        31, 32, 33, 0, 0,
        41, 42, 43, 0, 0,
        51, 52, 53, 0, 0);

    let expected3 = Matrix5::new(
        11, 12, 0, 0, 13,
        21, 22, 0, 0, 23,
        31, 32, 0, 0, 33,
        41, 42, 0, 0, 43,
        51, 52, 0, 0, 53);

    assert_eq!(m.insert_fixed_columns::<U2>(0, 0), expected1);
    assert_eq!(m.insert_fixed_columns::<U2>(3, 0), expected2);
    assert_eq!(m.insert_fixed_columns::<U2>(2, 0), expected3);

    // The following is just to verify that the return type dimensions is correctly inferred.
    let computed: Matrix<_, U5, Dynamic, _> = m.insert_columns(3, 2, 0);
    assert!(computed.eq(&expected2));
}


#[test]
fn insert_rows() {
    let m = Matrix3x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let expected1 = Matrix4x5::new(
         0,  0,  0,  0,  0,
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let expected2 = Matrix4x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
         0,  0,  0,  0,  0);

    let expected3 = Matrix4x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
         0,  0,  0,  0,  0,
        31, 32, 33, 34, 35);

    assert_eq!(m.insert_row(0, 0), expected1);
    assert_eq!(m.insert_row(3, 0), expected2);
    assert_eq!(m.insert_row(2, 0), expected3);

    let expected1 = Matrix5::new(
         0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let expected2 = Matrix5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
         0,  0,  0,  0,  0,
         0,  0,  0,  0,  0);

    let expected3 = Matrix5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
         0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,
        31, 32, 33, 34, 35);

    assert_eq!(m.insert_fixed_rows::<U2>(0, 0), expected1);
    assert_eq!(m.insert_fixed_rows::<U2>(3, 0), expected2);
    assert_eq!(m.insert_fixed_rows::<U2>(2, 0), expected3);

    // The following is just to verify that the return type dimensions is correctly inferred.
    let computed: Matrix<_, Dynamic, U5, _> = m.insert_rows(3, 2, 0);
    assert!(computed.eq(&expected2));
}

#[test]
fn resize() {
    let m = Matrix3x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let add_add = DMatrix::from_row_slice(5, 6, &[
        11, 12, 13, 14, 15, 42,
        21, 22, 23, 24, 25, 42,
        31, 32, 33, 34, 35, 42,
        42, 42, 42, 42, 42, 42,
        42, 42, 42, 42, 42, 42]);

    let del_del = DMatrix::from_row_slice(1, 2, &[11, 12]);

    let add_del = DMatrix::from_row_slice(5, 2, &[
        11, 12,
        21, 22,
        31, 32,
        42, 42,
        42, 42]);

    let del_add = DMatrix::from_row_slice(1, 8, &[
        11, 12, 13, 14, 15, 42, 42, 42]);

    assert_eq!(del_del, m.resize(1, 2, 42));
    assert_eq!(add_add, m.resize(5, 6, 42));
    assert_eq!(add_del, m.resize(5, 2, 42));
    assert_eq!(del_add, m.resize(1, 8, 42));
}
