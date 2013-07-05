use std::num::{Zero, One, Algebraic, Bounded};
use std::rand::{Rand, Rng, RngUtil};
use std::vec::{VecIterator, VecMutIterator};
use std::iterator::{Iterator, IteratorUtil, FromIterator};
use std::cmp::ApproxEq;
use std::uint::iterate;
use traits::iterable::{Iterable, IterableMut, FromAnyIterator};
use traits::basis::Basis;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::{Translation, Translatable};
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};
use traits::ring::Ring;
use traits::division_ring::DivisionRing;
use traits::homogeneous::{ToHomogeneous, FromHomogeneous};
use traits::indexable::Indexable;

mod vec_impl;


#[deriving(Ord, ToStr)]
pub struct Vec1<N>
{ at: [N, ..1] }

clone_impl!(Vec1)
deep_clone_impl!(Vec1)
new_impl!(Vec1, 1)
new_repeat_impl!(Vec1, elem, [elem])
indexable_impl!(Vec1)
dim_impl!(Vec1, 1)
eq_impl!(Vec1)
// (specialized) basis_impl!(Vec1, 1)
add_impl!(Vec1)
sub_impl!(Vec1)
neg_impl!(Vec1)
dot_impl!(Vec1, 1)
sub_dot_impl!(Vec1, 1)
scalar_mul_impl!(Vec1, 1)
scalar_div_impl!(Vec1, 1)
scalar_add_impl!(Vec1, 1)
scalar_sub_impl!(Vec1, 1)
translation_impl!(Vec1)
translatable_impl!(Vec1)
norm_impl!(Vec1, 1)
approx_eq_impl!(Vec1)
zero_impl!(Vec1)
one_impl!(Vec1)
rand_impl!(Vec1, rng, [rng])
from_iterator_impl!(Vec1, iterator, [iterator])
from_any_iterator_impl!(Vec1, iterator, [iterator])
bounded_impl!(Vec1)
iterable_impl!(Vec1)
iterable_mut_impl!(Vec1)
to_homogeneous_impl!(Vec1, Vec2)
from_homogeneous_impl!(Vec2, Vec1, 2)

#[deriving(Ord, ToStr)]
pub struct Vec2<N>
{ at: [N, ..2] }

clone_impl!(Vec2)
deep_clone_impl!(Vec2)
new_impl!(Vec2, 2)
new_repeat_impl!(Vec2, elem, [elem | elem])
indexable_impl!(Vec2)
dim_impl!(Vec2, 2)
eq_impl!(Vec2)
// (specialized) basis_impl!(Vec2, 2)
add_impl!(Vec2)
sub_impl!(Vec2)
neg_impl!(Vec2)
dot_impl!(Vec2, 2)
sub_dot_impl!(Vec2, 2)
scalar_mul_impl!(Vec2, 2)
scalar_div_impl!(Vec2, 2)
scalar_add_impl!(Vec2, 2)
scalar_sub_impl!(Vec2, 2)
translation_impl!(Vec2)
translatable_impl!(Vec2)
norm_impl!(Vec2, 2)
approx_eq_impl!(Vec2)
zero_impl!(Vec2)
one_impl!(Vec2)
rand_impl!(Vec2, rng, [rng | rng])
from_iterator_impl!(Vec2, iterator, [iterator | iterator])
from_any_iterator_impl!(Vec2, iterator, [iterator | iterator])
bounded_impl!(Vec2)
iterable_impl!(Vec2)
iterable_mut_impl!(Vec2)
to_homogeneous_impl!(Vec2, Vec3)
from_homogeneous_impl!(Vec3, Vec2, 3)

#[deriving(Ord, ToStr)]
pub struct Vec3<N>
{ at: [N, ..3] }

clone_impl!(Vec3)
deep_clone_impl!(Vec3)
new_impl!(Vec3, 3)
new_repeat_impl!(Vec3, elem, [elem | elem | elem])
indexable_impl!(Vec3)
dim_impl!(Vec3, 3)
eq_impl!(Vec3)
// (specialized) basis_impl!(Vec3, 3)
add_impl!(Vec3)
sub_impl!(Vec3)
neg_impl!(Vec3)
dot_impl!(Vec3, 3)
sub_dot_impl!(Vec3, 3)
scalar_mul_impl!(Vec3, 3)
scalar_div_impl!(Vec3, 3)
scalar_add_impl!(Vec3, 3)
scalar_sub_impl!(Vec3, 3)
translation_impl!(Vec3)
translatable_impl!(Vec3)
norm_impl!(Vec3, 3)
approx_eq_impl!(Vec3)
zero_impl!(Vec3)
one_impl!(Vec3)
rand_impl!(Vec3, rng, [rng | rng | rng])
from_iterator_impl!(Vec3, iterator, [iterator | iterator | iterator])
from_any_iterator_impl!(Vec3, iterator, [iterator | iterator | iterator])
bounded_impl!(Vec3)
iterable_impl!(Vec3)
iterable_mut_impl!(Vec3)
to_homogeneous_impl!(Vec3, Vec4)
from_homogeneous_impl!(Vec4, Vec3, 4)

#[deriving(Ord, ToStr)]
pub struct Vec4<N>
{ at: [N, ..4] }

clone_impl!(Vec4)
deep_clone_impl!(Vec4)
new_impl!(Vec4, 4)
new_repeat_impl!(Vec4, elem, [elem | elem | elem | elem])
indexable_impl!(Vec4)
dim_impl!(Vec4, 4)
eq_impl!(Vec4)
basis_impl!(Vec4, 4)
add_impl!(Vec4)
sub_impl!(Vec4)
neg_impl!(Vec4)
dot_impl!(Vec4, 4)
sub_dot_impl!(Vec4, 4)
scalar_mul_impl!(Vec4, 4)
scalar_div_impl!(Vec4, 4)
scalar_add_impl!(Vec4, 4)
scalar_sub_impl!(Vec4, 4)
translation_impl!(Vec4)
translatable_impl!(Vec4)
norm_impl!(Vec4, 4)
approx_eq_impl!(Vec4)
zero_impl!(Vec4)
one_impl!(Vec4)
rand_impl!(Vec4, rng, [rng | rng | rng | rng])
from_iterator_impl!(Vec4, iterator, [iterator | iterator | iterator | iterator])
from_any_iterator_impl!(Vec4, iterator, [iterator | iterator | iterator | iterator])
bounded_impl!(Vec4)
iterable_impl!(Vec4)
iterable_mut_impl!(Vec4)
to_homogeneous_impl!(Vec4, Vec5)
from_homogeneous_impl!(Vec5, Vec4, 5)

#[deriving(Ord, ToStr)]
pub struct Vec5<N>
{ at: [N, ..5] }

clone_impl!(Vec5)
deep_clone_impl!(Vec5)
new_impl!(Vec5, 5)
new_repeat_impl!(Vec5, elem, [elem | elem | elem | elem | elem])
indexable_impl!(Vec5)
dim_impl!(Vec5, 5)
eq_impl!(Vec5)
basis_impl!(Vec5, 5)
add_impl!(Vec5)
sub_impl!(Vec5)
neg_impl!(Vec5)
dot_impl!(Vec5, 5)
sub_dot_impl!(Vec5, 5)
scalar_mul_impl!(Vec5, 5)
scalar_div_impl!(Vec5, 5)
scalar_add_impl!(Vec5, 5)
scalar_sub_impl!(Vec5, 5)
translation_impl!(Vec5)
translatable_impl!(Vec5)
norm_impl!(Vec5, 5)
approx_eq_impl!(Vec5)
zero_impl!(Vec5)
one_impl!(Vec5)
rand_impl!(Vec5, rng, [rng | rng | rng | rng | rng])
from_iterator_impl!(Vec5, iterator, [iterator | iterator | iterator | iterator | iterator])
from_any_iterator_impl!(Vec5, iterator, [iterator | iterator | iterator | iterator | iterator])
bounded_impl!(Vec5)
iterable_impl!(Vec5)
iterable_mut_impl!(Vec5)
to_homogeneous_impl!(Vec5, Vec6)
from_homogeneous_impl!(Vec6, Vec5, 6)

#[deriving(Ord, ToStr)]
pub struct Vec6<N>
{ at: [N, ..6] }

clone_impl!(Vec6)
deep_clone_impl!(Vec6)
new_impl!(Vec6, 6)
new_repeat_impl!(Vec6, elem, [elem | elem | elem | elem | elem | elem])
indexable_impl!(Vec6)
dim_impl!(Vec6, 6)
eq_impl!(Vec6)
basis_impl!(Vec6, 6)
add_impl!(Vec6)
sub_impl!(Vec6)
neg_impl!(Vec6)
dot_impl!(Vec6, 6)
sub_dot_impl!(Vec6, 6)
scalar_mul_impl!(Vec6, 6)
scalar_div_impl!(Vec6, 6)
scalar_add_impl!(Vec6, 6)
scalar_sub_impl!(Vec6, 6)
translation_impl!(Vec6)
translatable_impl!(Vec6)
norm_impl!(Vec6, 6)
approx_eq_impl!(Vec6)
zero_impl!(Vec6)
one_impl!(Vec6)
rand_impl!(Vec6, rng, [rng | rng | rng | rng | rng | rng])
from_iterator_impl!(Vec6, iterator, [iterator | iterator | iterator | iterator | iterator | iterator])
from_any_iterator_impl!(Vec6, iterator, [iterator | iterator | iterator | iterator | iterator | iterator])
bounded_impl!(Vec6)
iterable_impl!(Vec6)
iterable_mut_impl!(Vec6)
