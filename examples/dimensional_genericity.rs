extern crate alga;
extern crate nalgebra as na;

use alga::linear::FiniteDimInnerSpace;
use na::{DefaultAllocator, Real, Unit, Vector2, Vector3, VectorN};
use na::allocator::Allocator;
use na::dimension::Dim;

/// Reflects a vector wrt. the hyperplane with normal `plane_normal`.
fn reflect_wrt_hyperplane_with_algebraic_genericity<V>(plane_normal: &Unit<V>, vector: &V) -> V
where
    V: FiniteDimInnerSpace + Copy,
{
    let n = plane_normal.as_ref(); // Get the underlying vector of type `V`.
    *vector - *n * (n.dot(vector) * na::convert(2.0))
}

/// Reflects a vector wrt. the hyperplane with normal `plane_normal`.
fn reflect_wrt_hyperplane_with_dimensional_genericity<N: Real, D: Dim>(
    plane_normal: &Unit<VectorN<N, D>>,
    vector: &VectorN<N, D>,
) -> VectorN<N, D>
where
    N: Real,
    D: Dim,
    DefaultAllocator: Allocator<N, D>,
{
    let n = plane_normal.as_ref(); // Get the underlying V.
    vector - n * (n.dot(vector) * na::convert(2.0))
}

/// Reflects a 2D vector wrt. the 2D line with normal `plane_normal`.
fn reflect_wrt_hyperplane2<N>(plane_normal: &Unit<Vector2<N>>, vector: &Vector2<N>) -> Vector2<N>
where
    N: Real,
{
    let n = plane_normal.as_ref(); // Get the underlying Vector2
    vector - n * (n.dot(vector) * na::convert(2.0))
}

/// Reflects a 3D vector wrt. the 3D plane with normal `plane_normal`.
/// /!\ This is an exact replicate of `reflect_wrt_hyperplane2, but for 3D.
fn reflect_wrt_hyperplane3<N>(plane_normal: &Unit<Vector3<N>>, vector: &Vector3<N>) -> Vector3<N>
where
    N: Real,
{
    let n = plane_normal.as_ref(); // Get the underlying Vector3
    vector - n * (n.dot(vector) * na::convert(2.0))
}

fn main() {
    let plane2 = Vector2::y_axis(); // 2D plane normal.
    let plane3 = Vector3::y_axis(); // 3D plane normal.

    let v2 = Vector2::new(1.0, 2.0); // 2D vector to be reflected.
    let v3 = Vector3::new(1.0, 2.0, 3.0); // 3D vector to be reflected.

    // We can call the same function for 2D and 3D.
    assert_eq!(
        reflect_wrt_hyperplane_with_algebraic_genericity(&plane2, &v2).y,
        -2.0
    );
    assert_eq!(
        reflect_wrt_hyperplane_with_algebraic_genericity(&plane3, &v3).y,
        -2.0
    );

    assert_eq!(
        reflect_wrt_hyperplane_with_dimensional_genericity(&plane2, &v2).y,
        -2.0
    );
    assert_eq!(
        reflect_wrt_hyperplane_with_dimensional_genericity(&plane3, &v3).y,
        -2.0
    );

    // Call each specific implementation depending on the dimension.
    assert_eq!(reflect_wrt_hyperplane2(&plane2, &v2).y, -2.0);
    assert_eq!(reflect_wrt_hyperplane3(&plane3, &v3).y, -2.0);
}
