extern crate nalgebra as na;
extern crate rand;

use rand::random;
use na::{Pnt2, Pnt3, Vec3, Vec1, Rot2, Rot3, Persp3, PerspMat3, Ortho3, OrthoMat3, Iso2, Iso3,
         Sim2, Sim3, BaseFloat, Transform};

macro_rules! test_transform_inv_transform_impl(
  ($t: ty, $p: ty) => (
    for _ in 0usize .. 10000 {
      let randmat: $t = random();
      let expected: $p = random();

      let computed = randmat.inv_transform(&randmat.transform(&expected));

      assert!(na::approx_eq(&computed, &expected));
    }
  );
);

#[test]
fn test_rotation2() {
    for _ in 0usize .. 10000 {
        let randmat: na::Rot2<f64> = na::one();
        let ang    = Vec1::new(na::abs(&random::<f64>()) % <f64 as BaseFloat>::pi());

        assert!(na::approx_eq(&na::rotation(&na::append_rotation(&randmat, &ang)), &ang));
    }
}

#[test]
fn test_inv_rotation3() {
    for _ in 0usize .. 10000 {
        let randmat: Rot3<f64> = na::one();
        let dir:     Vec3<f64> = random();
        let ang            = na::normalize(&dir) * (na::abs(&random::<f64>()) % <f64 as BaseFloat>::pi());
        let rot            = na::append_rotation(&randmat, &ang);

        assert!(na::approx_eq(&(na::transpose(&rot) * rot), &na::one()));
    }
}

#[test]
fn test_rot3_rotation_between() {
    let r1: Rot3<f64> = random();
    let r2: Rot3<f64> = random();

    let delta = na::rotation_between(&r1, &r2);

    assert!(na::approx_eq(&(delta * r1), &r2))
}

#[test]
fn test_rot3_angle_between() {
    let r1: Rot3<f64> = random();
    let r2: Rot3<f64> = random();

    let delta = na::rotation_between(&r1, &r2);
    let delta_angle = na::angle_between(&r1, &r2);

    assert!(na::approx_eq(&na::norm(&na::rotation(&delta)), &delta_angle))
}

#[test]
fn test_rot2_rotation_between() {
    let r1: Rot2<f64> = random();
    let r2: Rot2<f64> = random();

    let delta = na::rotation_between(&r1, &r2);

    assert!(na::approx_eq(&(delta * r1), &r2))
}

#[test]
fn test_rot2_angle_between() {
    let r1: Rot2<f64> = random();
    let r2: Rot2<f64> = random();

    let delta = na::rotation_between(&r1, &r2);
    let delta_angle = na::angle_between(&r1, &r2);

    assert!(na::approx_eq(&na::norm(&na::rotation(&delta)), &delta_angle))
}


#[test]
fn test_look_at_iso3() {
    for _ in 0usize .. 10000 {
        let eye     = random::<Pnt3<f64>>();
        let target  = random::<Pnt3<f64>>();
        let up      = random::<Vec3<f64>>();
        let viewmat = Iso3::new_look_at(&eye, &target, &up);

        assert_eq!(&(viewmat * eye), &na::orig());
        assert!(na::approx_eq(&na::normalize(&(viewmat * (target - eye))), &-Vec3::z()));
    }
}

#[test]
fn test_look_at_rot3() {
    for _ in 0usize .. 10000 {
        let dir     = random::<Vec3<f64>>();
        let up      = random::<Vec3<f64>>();
        let viewmat = Rot3::new_look_at(&dir, &up);

        assert!(na::approx_eq(&na::normalize(&(viewmat * dir)), &-Vec3::z()));
    }
}

#[test]
fn test_observer_frame_iso3() {
    for _ in 0usize .. 10000 {
        let eye      = random::<Pnt3<f64>>();
        let target   = random::<Pnt3<f64>>();
        let up       = random::<Vec3<f64>>();
        let observer = Iso3::new_observer_frame(&eye, &target, &up);

        assert_eq!(&(observer * na::orig::<Pnt3<f64>>()), &eye);
        assert!(na::approx_eq(&(observer * Vec3::z()), &na::normalize(&(target - eye))));
    }
}

#[test]
fn test_observer_frame_rot3() {
    for _ in 0usize .. 10000 {
        let dir      = random::<Vec3<f64>>();
        let up       = random::<Vec3<f64>>();
        let observer = Rot3::new_observer_frame(&dir, &up);

        assert!(na::approx_eq(&(observer * Vec3::z()), &na::normalize(&dir)));
    }
}

#[test]
fn test_persp() {
    let mut p  = Persp3::new(42.0f64, 0.5, 1.5, 10.0);
    let mut pm = PerspMat3::new(42.0f64, 0.5, 1.5, 10.0);
    assert!(p.to_mat() == pm.to_mat());
    assert!(p.aspect() == 42.0);
    assert!(p.fov()    == 0.5);
    assert!(p.znear()  == 1.5);
    assert!(p.zfar()   == 10.0);
    assert!(na::approx_eq(&pm.aspect(), &42.0));
    assert!(na::approx_eq(&pm.fov(),    &0.5));
    assert!(na::approx_eq(&pm.znear(),  &1.5));
    assert!(na::approx_eq(&pm.zfar(),   &10.0));

    p.set_fov(0.1);
    pm.set_fov(0.1);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_znear(24.0);
    pm.set_znear(24.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_zfar(61.0);
    pm.set_zfar(61.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_aspect(23.0);
    pm.set_aspect(23.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    assert!(p.aspect() == 23.0);
    assert!(p.fov()    == 0.1);
    assert!(p.znear()  == 24.0);
    assert!(p.zfar()   == 61.0);
    assert!(na::approx_eq(&pm.aspect(), &23.0));
    assert!(na::approx_eq(&pm.fov(),    &0.1));
    assert!(na::approx_eq(&pm.znear(),  &24.0));
    assert!(na::approx_eq(&pm.zfar(),   &61.0));
}

#[test]
fn test_ortho() {
    let mut p  = Ortho3::new(42.0f64, 0.5, 1.5, 10.0);
    let mut pm = OrthoMat3::new(42.0f64, 0.5, 1.5, 10.0);
    assert!(p.to_mat() == pm.to_mat());
    assert!(p.width()  == 42.0);
    assert!(p.height() == 0.5);
    assert!(p.znear()  == 1.5);
    assert!(p.zfar()   == 10.0);
    assert!(na::approx_eq(&pm.width(),  &42.0));
    assert!(na::approx_eq(&pm.height(), &0.5));
    assert!(na::approx_eq(&pm.znear(),  &1.5));
    assert!(na::approx_eq(&pm.zfar(),   &10.0));

    p.set_width(0.1);
    pm.set_width(0.1);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_znear(24.0);
    pm.set_znear(24.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_zfar(61.0);
    pm.set_zfar(61.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_height(23.0);
    pm.set_height(23.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    assert!(p.height() == 23.0);
    assert!(p.width()  == 0.1);
    assert!(p.znear()  == 24.0);
    assert!(p.zfar()   == 61.0);
    assert!(na::approx_eq(&pm.height(), &23.0));
    assert!(na::approx_eq(&pm.width(),  &0.1));
    assert!(na::approx_eq(&pm.znear(),  &24.0));
    assert!(na::approx_eq(&pm.zfar(),   &61.0));
}

#[test]
fn test_transform_inv_transform_rot2() {
    test_transform_inv_transform_impl!(Rot2<f32>, Pnt2<f32>);
}

#[test]
fn test_transform_inv_transform_rot3() {
    test_transform_inv_transform_impl!(Rot3<f32>, Pnt3<f32>);
}

#[test]
fn test_transform_inv_transform_iso2() {
    test_transform_inv_transform_impl!(Iso2<f32>, Pnt2<f32>);
}

#[test]
fn test_transform_inv_transform_iso3() {
    test_transform_inv_transform_impl!(Iso3<f32>, Pnt3<f32>);
}

#[test]
fn test_transform_inv_transform_sim2() {
    test_transform_inv_transform_impl!(Sim2<f32>, Pnt2<f32>);
}

#[test]
fn test_transform_inv_transform_sim3() {
    test_transform_inv_transform_impl!(Sim3<f32>, Pnt3<f32>);
}
