#![cfg(feature="arbitrary")]

extern crate nalgebra as na;
extern crate quickcheck;
extern crate rand;

use quickcheck::{Arbitrary, StdGen};
use na::*;


macro_rules! trivial_arb_test(
    ($t: ty, $name: ident) => (
        #[test]
        fn $name() {
            let mut g = StdGen::new(rand::thread_rng(), 100);
            let _: $t = Arbitrary::arbitrary(&mut g);
        }
    )
);

trivial_arb_test!(Vector1<f64>, arb_vec1);
trivial_arb_test!(Vector2<f64>, arb_vec2);
trivial_arb_test!(Vector3<f64>, arb_vec3);
trivial_arb_test!(Vector4<f64>, arb_vec4);
trivial_arb_test!(Vector5<f64>, arb_vec5);
trivial_arb_test!(Vector6<f64>, arb_vec6);

trivial_arb_test!(Point1<f64>, arb_point1);
trivial_arb_test!(Point2<f64>, arb_point2);
trivial_arb_test!(Point3<f64>, arb_point3);
trivial_arb_test!(Point4<f64>, arb_point4);
trivial_arb_test!(Point5<f64>, arb_point5);
trivial_arb_test!(Point6<f64>, arb_point6);

trivial_arb_test!(Matrix1<f64>, arb_mat1);
trivial_arb_test!(Matrix2<f64>, arb_mat2);
trivial_arb_test!(Matrix3<f64>, arb_mat3);
trivial_arb_test!(Matrix4<f64>, arb_mat4);
trivial_arb_test!(Matrix5<f64>, arb_mat5);
trivial_arb_test!(Matrix6<f64>, arb_mat6);

trivial_arb_test!(DVector1<f64>, arb_dvec1);
trivial_arb_test!(DVector2<f64>, arb_dvec2);
trivial_arb_test!(DVector3<f64>, arb_dvec3);
trivial_arb_test!(DVector4<f64>, arb_dvec4);
trivial_arb_test!(DVector5<f64>, arb_dvec5);
trivial_arb_test!(DVector6<f64>, arb_dvec6);

trivial_arb_test!(DMatrix<f64>, arb_dmatrix);
trivial_arb_test!(DVector<f64>, arb_dvector);

trivial_arb_test!(Quaternion<f64>, arb_quaternion);
trivial_arb_test!(UnitQuaternion<f64>, arb_unit_quaternion);

trivial_arb_test!(Isometry2<f64>, arb_iso2);
trivial_arb_test!(Isometry3<f64>, arb_iso3);

trivial_arb_test!(Rotation2<f64>, arb_rot2);
trivial_arb_test!(Rotation3<f64>, arb_rot3);

trivial_arb_test!(Orthographic3<f64>, arb_ortho3);
trivial_arb_test!(OrthographicMatrix3<f64>, arb_ortho_mat3);
trivial_arb_test!(Perspective3<f64>, arb_persp3);
trivial_arb_test!(PerspectiveMatrix3<f64>, arb_perspective_mat3);
