use core::num::{One, Zero};
use core::rand::{Rand, Rng, RngUtil};
use std::cmp::FuzzyEq;
use traits::workarounds::rlmul::{RMul, LMul};
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::rotation::Rotation;
use traits::delta_transform::DeltaTransform;
use dim1::vec1::{Vec1, vec1};
use dim2::mat2::{Mat2, mat2};
use dim3::mat3::{Mat3, mat3};
use dim3::vec3::{Vec3};

#[deriving(Eq)]
pub struct Rotmat<M>
{
  priv submat: M
}

impl<M: Copy> Rotmat<M>
{
  fn submat(&self) -> M
  { self.submat }
}

pub fn rotmat2<T: Copy + Trigonometric + Neg<T>>(angle: T) -> Rotmat<Mat2<T>>
{
  let coa = angle.cos();
  let sia = angle.sin();

  Rotmat
  { submat: mat2(coa, -sia, sia, coa) }
}

pub fn rotmat3<T: Copy + Trigonometric + Neg<T> + One + Sub<T, T> + Add<T, T> +
                  Mul<T, T>>
(axis: &Vec3<T>, angle: T) -> Rotmat<Mat3<T>>
{
  let _1        = One::one::<T>();
  let ux        = axis.x;
  let uy        = axis.y;
  let uz        = axis.z;
  let sqx       = ux * ux;
  let sqy       = uy * uy;
  let sqz       = uz * uz;
  let cos       = angle.cos();
  let one_m_cos = _1 - cos;
  let sin       = angle.sin();

  Rotmat {
    submat: mat3(
      (sqx + (_1 - sqx) * cos),
      (ux * uy * one_m_cos - uz * sin),
      (ux * uz * one_m_cos + uy * sin),

      (ux * uy * one_m_cos + uz * sin),
      (sqy + (_1 - sqy) * cos),
      (uy * uz * one_m_cos - ux * sin),

      (ux * uz * one_m_cos - uy * sin),
      (uy * uz * one_m_cos + ux * sin),
      (sqz + (_1 - sqz) * cos))
  }
}

impl<T: Quot<T, T> + Trigonometric + Neg<T> + Mul<T, T> + Add<T, T> + Copy>
Rotation<Vec1<T>> for Rotmat<Mat2<T>>
{
  fn rotation(&self) -> Vec1<T>
  { vec1(-(self.submat.m12 / self.submat.m11).atan()) }

  fn rotated(&self, rot: &Vec1<T>) -> Rotmat<Mat2<T>>
  { rotmat2(rot.x) * *self }

  fn rotate(&mut self, rot: &Vec1<T>)
  { *self = self.rotated(rot) }
}

impl<T: Quot<T, T> + Trigonometric + Neg<T> + Mul<T, T> + Add<T, T> + Copy +
        One + Sub<T, T>>
Rotation<(Vec3<T>, T)> for Rotmat<Mat3<T>>
{
  fn rotation(&self) -> (Vec3<T>, T)
  { fail!("Not yet implemented.") }

  fn rotated(&self, &(axis, angle): &(Vec3<T>, T)) -> Rotmat<Mat3<T>>
  { rotmat3(&axis, angle) * *self }

  fn rotate(&mut self, rot: &(Vec3<T>, T))
  { *self = self.rotated(rot) }
}

impl<T: Copy + Rand + Trigonometric + Neg<T>> Rand for Rotmat<Mat2<T>>
{
  fn rand<R: Rng>(rng: &R) -> Rotmat<Mat2<T>>
  { rotmat2(rng.gen()) }
}

impl<T: Copy + Rand + Trigonometric + Neg<T> + One + Sub<T, T> + Add<T, T> +
       Mul<T, T>>
Rand for Rotmat<Mat3<T>>
{
  fn rand<R: Rng>(rng: &R) -> Rotmat<Mat3<T>>
  { rotmat3(&rng.gen(), rng.gen()) }
}

impl<M: Dim> Dim for Rotmat<M>
{
  fn dim() -> uint
  { Dim::dim::<M>() }
}
 
impl<M: Copy + One + Zero> One for Rotmat<M>
{
  fn one() -> Rotmat<M>
  { Rotmat { submat: One::one() } }
}

impl<M: Copy + Mul<M, M>> Mul<Rotmat<M>, Rotmat<M>> for Rotmat<M>
{
  fn mul(&self, other: &Rotmat<M>) -> Rotmat<M>
  { Rotmat { submat: self.submat.mul(&other.submat) } }
}

impl<V, M: RMul<V>> RMul<V> for Rotmat<M>
{
  fn rmul(&self, other: &V) -> V
  { self.submat.rmul(other) }
}

impl<V, M: LMul<V>> LMul<V> for Rotmat<M>
{
  fn lmul(&self, other: &V) -> V
  { self.submat.lmul(other) }
}

impl<M: Copy> DeltaTransform<M> for Rotmat<M>
{
  fn delta_transform(&self) -> M
  { self.submat }
}

impl<M: Copy + Transpose> Inv for Rotmat<M>
{
  fn invert(&mut self)
  { self.transpose() }

  fn inverse(&self) -> Rotmat<M>
  { self.transposed() }
}

impl<M: Copy + Transpose>
Transpose for Rotmat<M>
{
  fn transposed(&self) -> Rotmat<M>
  { Rotmat { submat: self.submat.transposed() } }

  fn transpose(&mut self)
  { self.submat.transpose() }
}

impl<T, M: FuzzyEq<T>> FuzzyEq<T> for Rotmat<M>
{
  fn fuzzy_eq(&self, other: &Rotmat<M>) -> bool
  { self.submat.fuzzy_eq(&other.submat) }

  fn fuzzy_eq_eps(&self, other: &Rotmat<M>, epsilon: &T) -> bool
  { self.submat.fuzzy_eq_eps(&other.submat, epsilon) }
}

impl<M: ToStr> ToStr for Rotmat<M>
{
  fn to_str(&self) -> ~str
  { ~"Rotmat {" + " submat: " + self.submat.to_str() + " }" }
}
