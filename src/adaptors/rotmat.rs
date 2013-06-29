use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::ring::Ring;
use traits::division_ring::DivisionRing;
use traits::rlmul::{RMul, LMul};
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::rotation::{Rotation, Rotate, Rotatable};
use traits::transformation::{Transform}; // FIXME: implement Transformation and Transformable
use traits::homogeneous::ToHomogeneous;
use traits::indexable::Indexable;
use vec::Vec1;
use mat::{Mat2, Mat3};
use vec::Vec3;

#[deriving(Eq, ToStr)]
pub struct Rotmat<M>
{ priv submat: M }

pub fn rotmat2<N: Copy + Trigonometric + Neg<N>>(angle: N) -> Rotmat<Mat2<N>>
{
  let coa = angle.cos();
  let sia = angle.sin();

  Rotmat
  { submat: Mat2::new( [ copy coa, -sia, copy sia, copy coa ] ) }
}

pub fn rotmat3<N: Copy + Trigonometric + Ring>
(axis: &Vec3<N>, angle: N) -> Rotmat<Mat3<N>>
{
  let _1        = One::one::<N>();
  let ux        = copy axis.at[0];
  let uy        = copy axis.at[1];
  let uz        = copy axis.at[2];
  let sqx       = ux * ux;
  let sqy       = uy * uy;
  let sqz       = uz * uz;
  let cos       = angle.cos();
  let one_m_cos = _1 - cos;
  let sin       = angle.sin();

  Rotmat {
    submat: Mat3::new( [
      (sqx + (_1 - sqx) * cos),
      (ux * uy * one_m_cos - uz * sin),
      (ux * uz * one_m_cos + uy * sin),

      (ux * uy * one_m_cos + uz * sin),
      (sqy + (_1 - sqy) * cos),
      (uy * uz * one_m_cos - ux * sin),

      (ux * uz * one_m_cos - uy * sin),
      (uy * uz * one_m_cos + ux * sin),
      (sqz + (_1 - sqz) * cos) ] )
  }
}

impl<N: Trigonometric + DivisionRing + Copy>
Rotation<Vec1<N>> for Rotmat<Mat2<N>>
{
  #[inline]
  fn rotation(&self) -> Vec1<N>
  { Vec1::new([ -(self.submat.at((0, 1)) / self.submat.at((0, 0))).atan() ]) }

  #[inline]
  fn inv_rotation(&self) -> Vec1<N>
  { -self.rotation() }

  #[inline]
  fn rotate_by(&mut self, rot: &Vec1<N>)
  { *self = self.rotated(rot) }
}

impl<N: Trigonometric + DivisionRing + Copy>
Rotatable<Vec1<N>, Rotmat<Mat2<N>>> for Rotmat<Mat2<N>>
{
  #[inline]
  fn rotated(&self, rot: &Vec1<N>) -> Rotmat<Mat2<N>>
  { rotmat2(copy rot.at[0]) * *self }
}

impl<N: Copy + Trigonometric + DivisionRing>
Rotation<(Vec3<N>, N)> for Rotmat<Mat3<N>>
{
  #[inline]
  fn rotation(&self) -> (Vec3<N>, N)
  { fail!("Not yet implemented.") }
  #[inline]

  fn inv_rotation(&self) -> (Vec3<N>, N)
  { fail!("Not yet implemented.") }


  #[inline]
  fn rotate_by(&mut self, rot: &(Vec3<N>, N))
  { *self = self.rotated(rot) }
}

impl<N: Copy + Trigonometric + DivisionRing>
Rotatable<(Vec3<N>, N), Rotmat<Mat3<N>>> for Rotmat<Mat3<N>>
{
  #[inline]
  fn rotated(&self, &(axis, angle): &(Vec3<N>, N)) -> Rotmat<Mat3<N>>
  { rotmat3(&axis, angle) * *self }
}

impl<N: Copy + Rand + Trigonometric + Neg<N>> Rand for Rotmat<Mat2<N>>
{
  #[inline]
  fn rand<R: Rng>(rng: &mut R) -> Rotmat<Mat2<N>>
  { rotmat2(rng.gen()) }
}

impl<M: RMul<V> + LMul<V>, V> Rotate<V> for Rotmat<M>
{
  #[inline]
  fn rotate(&self, v: &V) -> V
  { self.rmul(v) }

  #[inline]
  fn inv_rotate(&self, v: &V) -> V
  { self.lmul(v) }
}

impl<M: RMul<V> + LMul<V>, V> Transform<V> for Rotmat<M>
{
  #[inline]
  fn transform_vec(&self, v: &V) -> V
  { self.rotate(v) }

  #[inline]
  fn inv_transform(&self, v: &V) -> V
  { self.inv_rotate(v) }
}

impl<N: Copy + Rand + Trigonometric + Ring>
Rand for Rotmat<Mat3<N>>
{
  #[inline]
  fn rand<R: Rng>(rng: &mut R) -> Rotmat<Mat3<N>>
  { rotmat3(&rng.gen(), rng.gen()) }
}

impl<M: Dim> Dim for Rotmat<M>
{
  #[inline]
  fn dim() -> uint
  { Dim::dim::<M>() }
}
 
impl<M: One + Zero> One for Rotmat<M>
{
  #[inline]
  fn one() -> Rotmat<M>
  { Rotmat { submat: One::one() } }
}

impl<M: Mul<M, M>> Mul<Rotmat<M>, Rotmat<M>> for Rotmat<M>
{
  #[inline]
  fn mul(&self, other: &Rotmat<M>) -> Rotmat<M>
  { Rotmat { submat: self.submat.mul(&other.submat) } }
}

impl<V, M: RMul<V>> RMul<V> for Rotmat<M>
{
  #[inline]
  fn rmul(&self, other: &V) -> V
  { self.submat.rmul(other) }
}

impl<V, M: LMul<V>> LMul<V> for Rotmat<M>
{
  #[inline]
  fn lmul(&self, other: &V) -> V
  { self.submat.lmul(other) }
}

impl<M: Transpose> Inv for Rotmat<M>
{
  #[inline]
  fn invert(&mut self)
  { self.transpose() }

  #[inline]
  fn inverse(&self) -> Rotmat<M>
  { self.transposed() }
}

impl<M: Transpose>
Transpose for Rotmat<M>
{
  #[inline]
  fn transposed(&self) -> Rotmat<M>
  { Rotmat { submat: self.submat.transposed() } }

  #[inline]
  fn transpose(&mut self)
  { self.submat.transpose() }
}

// we loose the info that we are a rotation matrix
impl<M: ToHomogeneous<M2>, M2> ToHomogeneous<M2> for Rotmat<M>
{
  fn to_homogeneous(&self) -> M2
  { self.submat.to_homogeneous() }
}

impl<N: ApproxEq<N>, M: ApproxEq<N>> ApproxEq<N> for Rotmat<M>
{
  #[inline]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline]
  fn approx_eq(&self, other: &Rotmat<M>) -> bool
  { self.submat.approx_eq(&other.submat) }

  #[inline]
  fn approx_eq_eps(&self, other: &Rotmat<M>, epsilon: &N) -> bool
  { self.submat.approx_eq_eps(&other.submat, epsilon) }
}
