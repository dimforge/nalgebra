use na::{Unit, Vector2, Vector3, Vector4};

// Antiparallel unit vectors used to return `self` for every `t`, breaking the documented
// endpoint contract slerp(a, b, 1) == b. See issue #657.
#[test]
fn unit_slerp_antiparallel_honors_endpoints_2d() {
    let a = Unit::new_normalize(Vector2::new(1.0, 0.0));
    let b = Unit::new_normalize(Vector2::new(-1.0, 0.0));

    assert_relative_eq!(a.slerp(&b, 0.0).into_inner(), a.into_inner());
    assert_relative_eq!(a.slerp(&b, 1.0).into_inner(), b.into_inner());
}

#[test]
fn unit_slerp_antiparallel_honors_endpoints_3d() {
    let a = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
    let b = Unit::new_normalize(Vector3::new(0.0, -1.0, 0.0));

    assert_relative_eq!(a.slerp(&b, 0.0).into_inner(), a.into_inner());
    assert_relative_eq!(a.slerp(&b, 1.0).into_inner(), b.into_inner());
}

// The interior path for antiparallel inputs is arbitrary but must stay on the unit sphere and
// move away from `self` (the old code stayed pinned to `self`). At t = 0.5 the midpoint is a
// quarter turn from both endpoints, hence orthogonal to `self`.
#[test]
fn unit_slerp_antiparallel_interior_is_unit_and_moves() {
    let a = Unit::new_normalize(Vector3::new(1.0, 2.0, -2.0));
    let b = Unit::new_normalize(-a.into_inner());

    for &t in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let m = a.slerp(&b, t).into_inner();
        assert_relative_eq!(m.norm(), 1.0);
    }
    let mid = a.slerp(&b, 0.5).into_inner();
    assert_relative_eq!(mid.dot(&a), 0.0, epsilon = 1.0e-12);
}

#[test]
fn unit_slerp_antiparallel_4d_endpoints() {
    let a = Unit::new_normalize(Vector4::new(1.0, -1.0, 2.0, 0.5));
    let b = Unit::new_normalize(-a.into_inner());

    assert_relative_eq!(a.slerp(&b, 0.0).into_inner(), a.into_inner());
    assert_relative_eq!(a.slerp(&b, 1.0).into_inner(), b.into_inner());
}

// The non-degenerate path must be untouched.
#[test]
fn unit_slerp_generic_endpoints() {
    let a = Unit::new_normalize(Vector2::new(1.0, 2.0));
    let b = Unit::new_normalize(Vector2::new(2.0, -3.0));

    assert_relative_eq!(a.slerp(&b, 0.0).into_inner(), a.into_inner());
    assert_relative_eq!(a.slerp(&b, 1.0).into_inner(), b.into_inner());
}

// The `Vector::slerp` wrapper normalizes and delegates, so it inherits the fix.
#[test]
fn vector_slerp_antiparallel_honors_endpoints() {
    let a = Vector3::new(0.0, 3.0, 0.0);
    let b = Vector3::new(0.0, -5.0, 0.0);

    assert_relative_eq!(a.slerp(&b, 1.0), b.normalize());
    assert_relative_eq!(a.slerp(&b, 0.0), a.normalize());
}
