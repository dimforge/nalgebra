use na::{Orthographic3, Perspective3, Point3};

#[test]
fn perspective_inverse() {
    let proj = Perspective3::new(800.0 / 600.0, 3.14 / 2.0, 1.0, 1000.0);
    let inv = proj.inverse();

    let id = inv * proj.into_inner();

    assert!(id.is_identity(1.0e-7));
}

#[test]
fn orthographic_inverse() {
    let proj = Orthographic3::new(1.0, 2.0, -3.0, -2.5, 10.0, 900.0);
    let inv = proj.inverse();

    let id = inv * proj.into_inner();

    assert!(id.is_identity(1.0e-7));
}

#[test]
fn perspective_matrix_point_transformation() {
    // https://github.com/rustsim/nalgebra/issues/640
    let proj = Perspective3::new(4.0 / 3.0, 90.0, 0.1, 100.0);
    let perspective_inv = proj.as_matrix().try_inverse().unwrap();
    let some_point = Point3::new(1.0, 2.0, 0.0);

    assert_eq!(
        perspective_inv.transform_point(&some_point),
        Point3::from_homogeneous(perspective_inv * some_point.coords.push(1.0)).unwrap()
    );
}

#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    use na::{Orthographic3, Perspective3, Point3};

    quickcheck! {
        fn perspective_project_unproject(pt: Point3<f64>) -> bool {
            let proj = Perspective3::new(800.0 / 600.0, 3.14 / 2.0, 1.0, 1000.0);

            let projected   = proj.project_point(&pt);
            let unprojected = proj.unproject_point(&projected);

            relative_eq!(pt, unprojected, epsilon = 1.0e-7)
        }

        fn orthographic_project_unproject(pt: Point3<f64>) -> bool {
            let proj = Orthographic3::new(1.0, 2.0, -3.0, -2.5, 10.0, 900.0);

            let projected   = proj.project_point(&pt);
            let unprojected = proj.unproject_point(&projected);

            relative_eq!(pt, unprojected, epsilon = 1.0e-7)
        }
    }
}
