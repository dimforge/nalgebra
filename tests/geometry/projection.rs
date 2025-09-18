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
    // https://github.com/dimforge/nalgebra/issues/640
    let proj = Perspective3::new(4.0 / 3.0, 90.0, 0.1, 100.0);
    let perspective_inv = proj.as_matrix().try_inverse().unwrap();
    let some_point = Point3::new(1.0, 2.0, 0.0);

    assert_eq!(
        perspective_inv.transform_point(&some_point),
        Point3::from_homogeneous(perspective_inv * some_point.coords.push(1.0)).unwrap()
    );
}

#[cfg(feature = "proptest-support")]
mod proptest_tests {
    use na::{Orthographic3, Perspective3, Point3};

    use crate::proptest::*;
    use proptest::{prop_assert, proptest};

    proptest! {
        #[test]
        fn perspective_project_unproject(pt in point3()) {
            let proj = Perspective3::new(800.0 / 600.0, 3.14 / 2.0, 1.0, 1000.0);

            let projected   = proj.project_point(&pt);
            let unprojected = proj.unproject_point(&projected);

            prop_assert!(relative_eq!(pt, unprojected, epsilon = 1.0e-7))
        }

        #[test]
        fn orthographic_project_unproject(pt in point3()) {
            let proj = Orthographic3::new(1.0, 2.0, -3.0, -2.5, 10.0, 900.0);

            let projected   = proj.project_point(&pt);
            let unprojected = proj.unproject_point(&projected);

            prop_assert!(relative_eq!(pt, unprojected, epsilon = 1.0e-7))
        }

        #[test]
        fn perspective_project_vector(pt in point3(), vec in vector2()) {
            let proj = Perspective3::new(800.0 / 600.0, 3.14 / 2.0, 1.0, 1000.0);

            let proj_pt = proj.project_point(&pt);
            let proj_vec = proj.project_vector(&vec.push(pt.z));
            let proj_pt2 = proj.project_point(&(pt + vec.push(0.0)));

            let proj_pt_plus_proj_vec = Point3::from((proj_pt.xy() + proj_vec.xy()).coords.push(proj_pt.z));

            prop_assert!(relative_eq!(proj_pt_plus_proj_vec, proj_pt2, epsilon = 1.0e-7))
        }

        #[test]
        fn orthographic_project_vector(pt in point3(), vec in vector3()) {
            let proj = Orthographic3::new(1.0, 2.0, -3.0, -2.5, 10.0, 900.0);

            let proj_pt = proj.project_point(&pt);
            let proj_vec = proj.project_vector(&vec);
            let proj_pt2 = proj.project_point(&(pt + vec));

            let proj_pt_plus_proj_vec = proj_pt + proj_vec;

            prop_assert!(relative_eq!(proj_pt_plus_proj_vec, proj_pt2, epsilon = 1.0e-7))
        }
    }
}
