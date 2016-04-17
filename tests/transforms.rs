extern crate nalgebra as na;
extern crate rand;

use rand::random;
use na::{Point2, Point3, Vector2, Vector3, Vector1, Rotation2, Rotation3, Perspective3, PerspectiveMatrix3, Orthographic3, OrthographicMatrix3,
         Isometry2, Isometry3, Similarity2, Similarity3, BaseFloat, Transform};

#[test]
fn test_rotation2() {
    for _ in 0usize .. 10000 {
        let randmatrix: na::Rotation2<f64> = na::one();
        let ang    = Vector1::new(na::abs(&random::<f64>()) % <f64 as BaseFloat>::pi());

        assert!(na::approx_eq(&na::rotation(&na::append_rotation(&randmatrix, &ang)), &ang));
    }
}

#[test]
fn test_inverse_rotation3() {
    for _ in 0usize .. 10000 {
        let randmatrix: Rotation3<f64> = na::one();
        let dir:     Vector3<f64> = random();
        let ang            = na::normalize(&dir) * (na::abs(&random::<f64>()) % <f64 as BaseFloat>::pi());
        let rotation            = na::append_rotation(&randmatrix, &ang);

        assert!(na::approx_eq(&(na::transpose(&rotation) * rotation), &na::one()));
    }
}

#[test]
fn test_rot3_rotation_between() {
    let r1: Rotation3<f64> = random();
    let r2: Rotation3<f64> = random();

    let delta = na::rotation_between(&r1, &r2);

    assert!(na::approx_eq(&(delta * r1), &r2))
}

#[test]
fn test_rot3_angle_between() {
    let r1: Rotation3<f64> = random();
    let r2: Rotation3<f64> = random();

    let delta = na::rotation_between(&r1, &r2);
    let delta_angle = na::angle_between(&r1, &r2);

    assert!(na::approx_eq(&na::norm(&na::rotation(&delta)), &delta_angle))
}

#[test]
fn test_rot2_rotation_between() {
    let r1: Rotation2<f64> = random();
    let r2: Rotation2<f64> = random();

    let delta = na::rotation_between(&r1, &r2);

    assert!(na::approx_eq(&(delta * r1), &r2))
}

#[test]
fn test_rot2_angle_between() {
    let r1: Rotation2<f64> = random();
    let r2: Rotation2<f64> = random();

    let delta = na::rotation_between(&r1, &r2);
    let delta_angle = na::angle_between(&r1, &r2);

    assert!(na::approx_eq(&na::norm(&na::rotation(&delta)), &delta_angle))
}


#[test]
fn test_look_at_rh_iso3() {
    for _ in 0usize .. 10000 {
        let eye     = random::<Point3<f64>>();
        let target  = random::<Point3<f64>>();
        let up      = random::<Vector3<f64>>();
        let viewmatrix = Isometry3::look_at_rh(&eye, &target, &up);

        let origin: Point3<f64> = na::origin();
        assert_eq!(&(viewmatrix * eye), &origin);
        assert!(na::approx_eq(&na::normalize(&(viewmatrix * (target - eye))), &-Vector3::z()));
    }
}

#[test]
fn test_look_at_rh_rot3() {
    for _ in 0usize .. 10000 {
        let dir     = random::<Vector3<f64>>();
        let up      = random::<Vector3<f64>>();
        let viewmatrix = Rotation3::look_at_rh(&dir, &up);

        println!("found: {}", viewmatrix * dir);
        assert!(na::approx_eq(&na::normalize(&(viewmatrix * dir)), &-Vector3::z()));
    }
}

#[test]
fn test_observer_frame_iso3() {
    for _ in 0usize .. 10000 {
        let eye      = random::<Point3<f64>>();
        let target   = random::<Point3<f64>>();
        let up       = random::<Vector3<f64>>();
        let observer = Isometry3::new_observer_frame(&eye, &target, &up);

        assert_eq!(&(observer * na::origin::<Point3<f64>>()), &eye);
        assert!(na::approx_eq(&(observer * Vector3::z()), &na::normalize(&(target - eye))));
    }
}

#[test]
fn test_observer_frame_rot3() {
    for _ in 0usize .. 10000 {
        let dir      = random::<Vector3<f64>>();
        let up       = random::<Vector3<f64>>();
        let observer = Rotation3::new_observer_frame(&dir, &up);

        assert!(na::approx_eq(&(observer * Vector3::z()), &na::normalize(&dir)));
    }
}

#[test]
fn test_persp() {
    let mut p  = Perspective3::new(42.0f64, 0.5, 1.5, 10.0);
    let mut pm = PerspectiveMatrix3::new(42.0f64, 0.5, 1.5, 10.0);
    assert!(p.to_matrix() == pm.to_matrix());
    assert!(p.aspect() == 42.0);
    assert!(p.fovy()   == 0.5);
    assert!(p.znear()  == 1.5);
    assert!(p.zfar()   == 10.0);
    assert!(na::approx_eq(&pm.aspect(), &42.0));
    assert!(na::approx_eq(&pm.fovy(),   &0.5));
    assert!(na::approx_eq(&pm.znear(),  &1.5));
    assert!(na::approx_eq(&pm.zfar(),   &10.0));

    p.set_fovy(0.1);
    pm.set_fovy(0.1);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    p.set_znear(24.0);
    pm.set_znear(24.0);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    p.set_zfar(61.0);
    pm.set_zfar(61.0);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    p.set_aspect(23.0);
    pm.set_aspect(23.0);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    assert!(p.aspect() == 23.0);
    assert!(p.fovy()   == 0.1);
    assert!(p.znear()  == 24.0);
    assert!(p.zfar()   == 61.0);
    assert!(na::approx_eq(&pm.aspect(), &23.0));
    assert!(na::approx_eq(&pm.fovy(),   &0.1));
    assert!(na::approx_eq(&pm.znear(),  &24.0));
    assert!(na::approx_eq(&pm.zfar(),   &61.0));
}

#[test]
fn test_ortho() {
    let mut p  = Orthographic3::new(-0.3, 5.2, -3.9, -1.0, 1.5, 10.0);
    let mut pm = OrthographicMatrix3::new(-0.3, 5.2, -3.9, -1.0, 1.5, 10.0);
    assert!(p.to_matrix() == pm.to_matrix());
    assert!(p.left()   == -0.3);
    assert!(p.right()  == 5.2);
    assert!(p.bottom() == -3.9);
    assert!(p.top()    == -1.0);
    assert!(p.znear()  == 1.5);
    assert!(p.zfar()   == 10.0);
    assert!(na::approx_eq(&pm.left(),   &-0.3));
    assert!(na::approx_eq(&pm.right(),  &5.2));
    assert!(na::approx_eq(&pm.bottom(), &-3.9));
    assert!(na::approx_eq(&pm.top(),    &-1.0));
    assert!(na::approx_eq(&pm.znear(),  &1.5));
    assert!(na::approx_eq(&pm.zfar(),   &10.0));

    p.set_left(0.1);
    pm.set_left(0.1);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    p.set_right(10.1);
    pm.set_right(10.1);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    p.set_top(24.0);
    pm.set_top(24.0);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    p.set_bottom(-23.0);
    pm.set_bottom(-23.0);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    p.set_zfar(61.0);
    pm.set_zfar(61.0);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    p.set_znear(21.0);
    pm.set_znear(21.0);
    assert!(na::approx_eq(&p.to_matrix(), pm.as_matrix()));

    assert!(p.znear()  == 21.0);
    assert!(p.zfar()   == 61.0);
    assert!(na::approx_eq(&pm.znear(),  &21.0));
    assert!(na::approx_eq(&pm.zfar(),   &61.0));
}

macro_rules! test_transform_inverse_transform_impl(
  ($fnname: ident, $t: ty, $p: ty) => (
    #[test]
    fn $fnname() {
        for _ in 0usize .. 10000 {
          let randmatrix: $t = random();
          let expected: $p = random();
    
          let computed = randmatrix.inverse_transform(&randmatrix.transform(&expected));
          println!("computed: {}, expected: {}", computed, expected);
    
          assert!(na::approx_eq(&computed, &expected));
        }
    }
  );
);

test_transform_inverse_transform_impl!(test_transform_inverse_transform_rot2, Rotation2<f64>, Point2<f64>);
test_transform_inverse_transform_impl!(test_transform_inverse_transform_rot3, Rotation3<f64>, Point3<f64>);
test_transform_inverse_transform_impl!(test_transform_inverse_transform_iso2, Isometry2<f64>, Point2<f64>);
test_transform_inverse_transform_impl!(test_transform_inverse_transform_iso3, Isometry3<f64>, Point3<f64>);
test_transform_inverse_transform_impl!(test_transform_inverse_transform_sim2, Similarity2<f64>, Point2<f64>);
test_transform_inverse_transform_impl!(test_transform_inverse_transform_sim3, Similarity3<f64>, Point3<f64>);

macro_rules! test_transform_mul_assoc(
  ($fnname: ident, $t1: ty, $t2: ty, $p: ty) => (
    #[test]
    fn $fnname() {
        for _ in 0usize .. 10000 {
          let t1: $t1 = random();
          let t2: $t2 = random();
          let p:  $p  = random();

          let t1p  = t1 * p;
          let t2p  = t2 * p;
          let t1t2 = t1 * t2;
          let t2t1 = t2 * t1;
    
          assert!(na::approx_eq(&(t1t2 * p), &(t1 * t2p)));
          assert!(na::approx_eq(&(t2t1 * p), &(t2 * t1p)));
        }
    }
  );
);

test_transform_mul_assoc!(test_transform_inverse_transform_sim3_sim3_point3, Similarity3<f64>, Similarity3<f64>, Point3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim3_iso3_point3, Similarity3<f64>, Isometry3<f64>, Point3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim3_rot3_point3, Similarity3<f64>, Rotation3<f64>, Point3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_iso3_iso3_point3, Isometry3<f64>, Isometry3<f64>, Point3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_iso3_rot3_point3, Isometry3<f64>, Rotation3<f64>, Point3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_rot3_rot3_point3, Rotation3<f64>, Rotation3<f64>, Point3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim3_sim3_vec3, Similarity3<f64>, Similarity3<f64>, Vector3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim3_iso3_vec3, Similarity3<f64>, Isometry3<f64>, Vector3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim3_rot3_vec3, Similarity3<f64>, Rotation3<f64>, Vector3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_iso3_iso3_vec3, Isometry3<f64>, Isometry3<f64>, Vector3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_iso3_rot3_vec3, Isometry3<f64>, Rotation3<f64>, Vector3<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_rot3_rot3_vec3, Rotation3<f64>, Rotation3<f64>, Vector3<f64>);

test_transform_mul_assoc!(test_transform_inverse_transform_sim2_sim2_point2, Similarity2<f64>, Similarity2<f64>, Point2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim2_iso2_point2, Similarity2<f64>, Isometry2<f64>, Point2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim2_rot2_point2, Similarity2<f64>, Rotation2<f64>, Point2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_iso2_iso2_point2, Isometry2<f64>, Isometry2<f64>, Point2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_iso2_rot2_point2, Isometry2<f64>, Rotation2<f64>, Point2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_rot2_rot2_point2, Rotation2<f64>, Rotation2<f64>, Point2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim2_sim2_vec2, Similarity2<f64>, Similarity2<f64>, Vector2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim2_iso2_vec2, Similarity2<f64>, Isometry2<f64>, Vector2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_sim2_rot2_vec2, Similarity2<f64>, Rotation2<f64>, Vector2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_iso2_iso2_vec2, Isometry2<f64>, Isometry2<f64>, Vector2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_iso2_rot2_vec2, Isometry2<f64>, Rotation2<f64>, Vector2<f64>);
test_transform_mul_assoc!(test_transform_inverse_transform_rot2_rot2_vec2, Rotation2<f64>, Rotation2<f64>, Vector2<f64>);
