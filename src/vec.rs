use std::cast;
use std::num::{Zero, One, Algebraic, Bounded};
use std::rand::Rng;
use std::vec::{VecIterator, VecMutIterator};
use std::iter::{Iterator, FromIterator};
use std::cmp::ApproxEq;

use traits::translation::{Translation, Translate};
use traits::transformation::Transform;
use traits::rotation::Rotate;

pub use traits::homogeneous::{FromHomogeneous, ToHomogeneous};
pub use traits::vec_cast::VecCast;
pub use traits::vector::{Vec, VecExt, AlgebraicVec, AlgebraicVecExt};
pub use traits::basis::Basis;
pub use traits::dim::Dim;
pub use traits::indexable::Indexable;
pub use traits::iterable::{Iterable, IterableMut};
pub use traits::sample::UniformSphereSample;
pub use traits::scalar_op::{ScalarAdd, ScalarSub};
pub use traits::cross::{Cross, CrossMatrix};
pub use traits::outer::Outer;
pub use traits::dot::Dot;
pub use traits::norm::Norm;

// structs
pub use dvec::DVec;

mod vec_macros;

/// Vector of dimension 0.
#[deriving(Eq, Encodable, Decodable, Clone, DeepClone, Rand, Zero, ToStr)]
pub struct Vec0<N>;

/// Vector of dimension 1.
#[deriving(Eq, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec1<N> {
    /// First component of the vector.
    x: N
}

new_impl!(Vec1, x)
ord_impl!(Vec1, x)
orderable_impl!(Vec1, x)
vec_axis_impl!(Vec1, x)
vec_cast_impl!(Vec1, x)
indexable_impl!(Vec1, 1)
new_repeat_impl!(Vec1, val, x)
dim_impl!(Vec1, 1)
container_impl!(Vec1)
// (specialized) basis_impl!(Vec1, 1)
add_impl!(Vec1, x)
sub_impl!(Vec1, x)
neg_impl!(Vec1, x)
dot_impl!(Vec1, x)
scalar_mul_impl!(Vec1, x)
scalar_div_impl!(Vec1, x)
scalar_add_impl!(Vec1, x)
scalar_sub_impl!(Vec1, x)
translation_impl!(Vec1)
norm_impl!(Vec1)
approx_eq_impl!(Vec1, x)
round_impl!(Vec1, x)
one_impl!(Vec1)
from_iterator_impl!(Vec1, iterator)
bounded_impl!(Vec1)
iterable_impl!(Vec1, 1)
iterable_mut_impl!(Vec1, 1)
to_homogeneous_impl!(Vec1, Vec2, y, x)
from_homogeneous_impl!(Vec1, Vec2, y, x)
translate_impl!(Vec1)
rotate_impl!(Vec1)
transform_impl!(Vec1)

/// Vector of dimension 2.
#[deriving(Eq, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec2<N> {
    /// First component of the vector.
    x: N,
    /// Second component of the vector.
    y: N
}

new_impl!(Vec2, x, y)
ord_impl!(Vec2, x, y)
orderable_impl!(Vec2, x, y)
vec_axis_impl!(Vec2, x, y)
vec_cast_impl!(Vec2, x, y)
indexable_impl!(Vec2, 2)
new_repeat_impl!(Vec2, val, x, y)
dim_impl!(Vec2, 2)
container_impl!(Vec2)
// (specialized) basis_impl!(Vec2, 1)
add_impl!(Vec2, x, y)
sub_impl!(Vec2, x, y)
neg_impl!(Vec2, x, y)
dot_impl!(Vec2, x, y)
scalar_mul_impl!(Vec2, x, y)
scalar_div_impl!(Vec2, x, y)
scalar_add_impl!(Vec2, x, y)
scalar_sub_impl!(Vec2, x, y)
translation_impl!(Vec2)
norm_impl!(Vec2)
approx_eq_impl!(Vec2, x, y)
round_impl!(Vec2, x, y)
one_impl!(Vec2)
from_iterator_impl!(Vec2, iterator, iterator)
bounded_impl!(Vec2)
iterable_impl!(Vec2, 2)
iterable_mut_impl!(Vec2, 2)
to_homogeneous_impl!(Vec2, Vec3, z, x, y)
from_homogeneous_impl!(Vec2, Vec3, z, x, y)
translate_impl!(Vec2)
rotate_impl!(Vec2)
transform_impl!(Vec2)

/// Vector of dimension 3.
#[deriving(Eq, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec3<N> {
    /// First component of the vector.
    x: N,
    /// Second component of the vector.
    y: N,
    /// Third component of the vector.
    z: N
}

new_impl!(Vec3, x, y, z)
ord_impl!(Vec3, x, y, z)
orderable_impl!(Vec3, x, y, z)
vec_axis_impl!(Vec3, x, y, z)
vec_cast_impl!(Vec3, x, y, z)
indexable_impl!(Vec3, 3)
new_repeat_impl!(Vec3, val, x, y, z)
dim_impl!(Vec3, 3)
container_impl!(Vec3)
// (specialized) basis_impl!(Vec3, 1)
add_impl!(Vec3, x, y, z)
sub_impl!(Vec3, x, y, z)
neg_impl!(Vec3, x, y, z)
dot_impl!(Vec3, x, y, z)
scalar_mul_impl!(Vec3, x, y, z)
scalar_div_impl!(Vec3, x, y, z)
scalar_add_impl!(Vec3, x, y, z)
scalar_sub_impl!(Vec3, x, y, z)
translation_impl!(Vec3)
norm_impl!(Vec3)
approx_eq_impl!(Vec3, x, y, z)
round_impl!(Vec3, x, y, z)
one_impl!(Vec3)
from_iterator_impl!(Vec3, iterator, iterator, iterator)
bounded_impl!(Vec3)
iterable_impl!(Vec3, 3)
iterable_mut_impl!(Vec3, 3)
to_homogeneous_impl!(Vec3, Vec4, w, x, y, z)
from_homogeneous_impl!(Vec3, Vec4, w, x, y, z)
translate_impl!(Vec3)
rotate_impl!(Vec3)
transform_impl!(Vec3)


/// Vector of dimension 3 with an extra component for padding.
#[deriving(Eq, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct PVec3<N> {
    /// First component of the vector.
    x: N,
    /// Second component of the vector.
    y: N,
    /// Third component of the vector.
    z: N,
    // Unused component, for padding
    priv _unused: N
}

impl<N: Clone> PVec3<N> {
    /// Creates a new 3d vector.
    pub fn new(x: N, y: N, z: N) -> PVec3<N> {
        PVec3 { x: x.clone(), y: y, z: z, _unused: x }
    }
}

ord_impl!(PVec3, x, y, z)
orderable_impl!(PVec3, x, y, z)
vec_axis_impl!(PVec3, x, y, z)
vec_cast_impl!(PVec3, x, y, z)
indexable_impl!(PVec3, 3)
new_repeat_impl!(PVec3, val, x, y, z, _unused)
dim_impl!(PVec3, 3)
container_impl!(PVec3)
// (specialized) basis_impl!(PVec3, 1)
add_impl!(PVec3, x, y, z)
sub_impl!(PVec3, x, y, z)
neg_impl!(PVec3, x, y, z)
dot_impl!(PVec3, x, y, z)
scalar_mul_impl!(PVec3, x, y, z)
scalar_div_impl!(PVec3, x, y, z)
scalar_add_impl!(PVec3, x, y, z)
scalar_sub_impl!(PVec3, x, y, z)
translation_impl!(PVec3)
norm_impl!(PVec3)
approx_eq_impl!(PVec3, x, y, z)
round_impl!(PVec3, x, y, z)
one_impl!(PVec3)
from_iterator_impl!(PVec3, iterator, iterator, iterator)
bounded_impl!(PVec3)
iterable_impl!(PVec3, 3)
iterable_mut_impl!(PVec3, 3)
to_homogeneous_impl!(PVec3, Vec4, w, x, y, z)
from_homogeneous_impl!(PVec3, Vec4, w, x, y, z)
translate_impl!(PVec3)
rotate_impl!(PVec3)
transform_impl!(PVec3)



/// Vector of dimension 4.
#[deriving(Eq, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec4<N> {
    /// First component of the vector.
    x: N,
    /// Second component of the vector.
    y: N,
    /// Third component of the vector.
    z: N,
    /// Fourth component of the vector.
    w: N
}

new_impl!(Vec4, x, y, z, w)
ord_impl!(Vec4, x, y, z, w)
orderable_impl!(Vec4, x, y, z, w)
vec_axis_impl!(Vec4, x, y, z, w)
vec_cast_impl!(Vec4, x, y, z, w)
indexable_impl!(Vec4, 4)
new_repeat_impl!(Vec4, val, x, y, z, w)
dim_impl!(Vec4, 4)
container_impl!(Vec4)
basis_impl!(Vec4, 4)
add_impl!(Vec4, x, y, z, w)
sub_impl!(Vec4, x, y, z, w)
neg_impl!(Vec4, x, y, z, w)
dot_impl!(Vec4, x, y, z, w)
scalar_mul_impl!(Vec4, x, y, z, w)
scalar_div_impl!(Vec4, x, y, z, w)
scalar_add_impl!(Vec4, x, y, z, w)
scalar_sub_impl!(Vec4, x, y, z, w)
translation_impl!(Vec4)
norm_impl!(Vec4)
approx_eq_impl!(Vec4, x, y, z, w)
round_impl!(Vec4, x, y, z, w)
one_impl!(Vec4)
from_iterator_impl!(Vec4, iterator, iterator, iterator, iterator)
bounded_impl!(Vec4)
iterable_impl!(Vec4, 4)
iterable_mut_impl!(Vec4, 4)
to_homogeneous_impl!(Vec4, Vec5, a, x, y, z, w)
from_homogeneous_impl!(Vec4, Vec5, a, x, y, z, w)
translate_impl!(Vec4)
rotate_impl!(Vec4)
transform_impl!(Vec4)

/// Vector of dimension 5.
#[deriving(Eq, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec5<N> {
    /// First component of the vector.
    x: N,
    /// Second component of the vector.
    y: N,
    /// Third component of the vector.
    z: N,
    /// Fourth component of the vector.
    w: N,
    /// Fifth of the vector.
    a: N
}

new_impl!(Vec5, x, y, z, w, a)
ord_impl!(Vec5, x, y, z, w, a)
orderable_impl!(Vec5, x, y, z, w, a)
vec_axis_impl!(Vec5, x, y, z, w, a)
vec_cast_impl!(Vec5, x, y, z, w, a)
indexable_impl!(Vec5, 5)
new_repeat_impl!(Vec5, val, x, y, z, w, a)
dim_impl!(Vec5, 5)
container_impl!(Vec5)
basis_impl!(Vec5, 5)
add_impl!(Vec5, x, y, z, w, a)
sub_impl!(Vec5, x, y, z, w, a)
neg_impl!(Vec5, x, y, z, w, a)
dot_impl!(Vec5, x, y, z, w, a)
scalar_mul_impl!(Vec5, x, y, z, w, a)
scalar_div_impl!(Vec5, x, y, z, w, a)
scalar_add_impl!(Vec5, x, y, z, w, a)
scalar_sub_impl!(Vec5, x, y, z, w, a)
translation_impl!(Vec5)
norm_impl!(Vec5)
approx_eq_impl!(Vec5, x, y, z, w, a)
round_impl!(Vec5, x, y, z, w, a)
one_impl!(Vec5)
from_iterator_impl!(Vec5, iterator, iterator, iterator, iterator, iterator)
bounded_impl!(Vec5)
iterable_impl!(Vec5, 5)
iterable_mut_impl!(Vec5, 5)
to_homogeneous_impl!(Vec5, Vec6, b, x, y, z, w, a)
from_homogeneous_impl!(Vec5, Vec6, b, x, y, z, w, a)
translate_impl!(Vec5)
rotate_impl!(Vec5)
transform_impl!(Vec5)

/// Vector of dimension 6.
#[deriving(Eq, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec6<N> {
    /// First component of the vector.
    x: N,
    /// Second component of the vector.
    y: N,
    /// Third component of the vector.
    z: N,
    /// Fourth component of the vector.
    w: N,
    /// Fifth of the vector.
    a: N,
    /// Sixth component of the vector.
    b: N
}

new_impl!(Vec6, x, y, z, w, a, b)
ord_impl!(Vec6, x, y, z, w, a, b)
orderable_impl!(Vec6, x, y, z, w, a, b)
vec_axis_impl!(Vec6, x, y, z, w, a, b)
vec_cast_impl!(Vec6, x, y, z, w, a, b)
indexable_impl!(Vec6, 6)
new_repeat_impl!(Vec6, val, x, y, z, w, a, b)
dim_impl!(Vec6, 6)
container_impl!(Vec6)
basis_impl!(Vec6, 6)
add_impl!(Vec6, x, y, z, w, a, b)
sub_impl!(Vec6, x, y, z, w, a, b)
neg_impl!(Vec6, x, y, z, w, a, b)
dot_impl!(Vec6, x, y, z, w, a, b)
scalar_mul_impl!(Vec6, x, y, z, w, a, b)
scalar_div_impl!(Vec6, x, y, z, w, a, b)
scalar_add_impl!(Vec6, x, y, z, w, a, b)
scalar_sub_impl!(Vec6, x, y, z, w, a, b)
translation_impl!(Vec6)
norm_impl!(Vec6)
approx_eq_impl!(Vec6, x, y, z, w, a, b)
round_impl!(Vec6, x, y, z, w, a, b)
one_impl!(Vec6)
from_iterator_impl!(Vec6, iterator, iterator, iterator, iterator, iterator, iterator)
bounded_impl!(Vec6)
iterable_impl!(Vec6, 6)
iterable_mut_impl!(Vec6, 6)
translate_impl!(Vec6)
rotate_impl!(Vec6)
transform_impl!(Vec6)
