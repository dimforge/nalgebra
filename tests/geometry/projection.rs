use na::{Perspective3, Orthographic3};

#[test]
fn perspective_inverse() {
    let proj = Perspective3::new(800.0 / 600.0, 3.14 / 2.0, 1.0, 1000.0);
    let inv  = proj.inverse();

    let id = inv * proj.unwrap();

    assert!(id.is_identity(1.0e-7));
}

#[test]
fn orthographic_inverse() {
    let proj = Orthographic3::new(1.0, 2.0, -3.0, -2.5, 10.0, 900.0);
    let inv  = proj.inverse();

    let id = inv * proj.unwrap();

    assert!(id.is_identity(1.0e-7));
}


#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    use na::{Point3, Perspective3, Orthographic3};

    quickcheck!{
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
