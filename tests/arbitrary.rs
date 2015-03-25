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

trivial_arb_test!(Vec1<f64>, arb_vec1);
trivial_arb_test!(Vec2<f64>, arb_vec2);
trivial_arb_test!(Vec3<f64>, arb_vec3);
trivial_arb_test!(Vec4<f64>, arb_vec4);
trivial_arb_test!(Vec5<f64>, arb_vec5);
trivial_arb_test!(Vec6<f64>, arb_vec6);

trivial_arb_test!(Pnt1<f64>, arb_pnt1);
trivial_arb_test!(Pnt2<f64>, arb_pnt2);
trivial_arb_test!(Pnt3<f64>, arb_pnt3);
trivial_arb_test!(Pnt4<f64>, arb_pnt4);
trivial_arb_test!(Pnt5<f64>, arb_pnt5);
trivial_arb_test!(Pnt6<f64>, arb_pnt6);

trivial_arb_test!(Mat1<f64>, arb_mat1);
trivial_arb_test!(Mat2<f64>, arb_mat2);
trivial_arb_test!(Mat3<f64>, arb_mat3);
trivial_arb_test!(Mat4<f64>, arb_mat4);
trivial_arb_test!(Mat5<f64>, arb_mat5);
trivial_arb_test!(Mat6<f64>, arb_mat6);

trivial_arb_test!(DVec1<f64>, arb_dvec1);
trivial_arb_test!(DVec2<f64>, arb_dvec2);
trivial_arb_test!(DVec3<f64>, arb_dvec3);
trivial_arb_test!(DVec4<f64>, arb_dvec4);
trivial_arb_test!(DVec5<f64>, arb_dvec5);
trivial_arb_test!(DVec6<f64>, arb_dvec6);

trivial_arb_test!(DMat<f64>, arb_dmat);
trivial_arb_test!(DVec<f64>, arb_dvec);

trivial_arb_test!(Quat<f64>, arb_quat);
trivial_arb_test!(UnitQuat<f64>, arb_unit_quat);

trivial_arb_test!(Iso2<f64>, arb_iso2);
trivial_arb_test!(Iso3<f64>, arb_iso3);

trivial_arb_test!(Rot2<f64>, arb_rot2);
trivial_arb_test!(Rot3<f64>, arb_rot3);

trivial_arb_test!(Ortho3<f64>, arb_ortho3);
trivial_arb_test!(OrthoMat3<f64>, arb_ortho_mat3);
trivial_arb_test!(Persp3<f64>, arb_persp3);
trivial_arb_test!(PerspMat3<f64>, arb_persp_mat3);
