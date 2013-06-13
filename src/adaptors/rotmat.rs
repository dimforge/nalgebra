use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::workarounds::rlmul::{RMul, LMul};
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::rotation::Rotation;
use traits::delta_transform::{DeltaTransform, DeltaTransformVector};
use dim1::vec1::Vec1;
use dim2::mat2::Mat2;
use dim3::mat3::Mat3;
use dim3::vec3::{Vec3};

#[deriving(Eq, ToStr)]
pub struct Rotmat<M>
{
  priv submat: M
}

impl<M: Copy> Rotmat<M>
{
  #[inline(always)]
  fn submat(&self) -> M
  { self.submat }
}

pub fn rotmat2<N: Copy + Trigonometric + Neg<N>>(angle: N) -> Rotmat<Mat2<N>>
{
  let coa = angle.cos();
  let sia = angle.sin();

  Rotmat
  { submat: Mat2::new(coa, -sia, sia, coa) }
}

pub fn rotmat3<N: Copy + Trigonometric + Neg<N> + One + Sub<N, N> + Add<N, N> +
                  Mul<N, N>>
(axis: &Vec3<N>, angle: N) -> Rotmat<Mat3<N>>
{
  let _1        = One::one::<N>();
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
    submat: Mat3::new(
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

impl<N: Div<N, N> + Trigonometric + Neg<N> + Mul<N, N> + Add<N, N> + Copy>
Rotation<Vec1<N>> for Rotmat<Mat2<N>>
{
  #[inline(always)]
  fn rotation(&self) -> Vec1<N>
  { Vec1::new(-(self.submat.m12 / self.submat.m11).atan()) }

  #[inline(always)]
  fn rotated(&self, rot: &Vec1<N>) -> Rotmat<Mat2<N>>
  { rotmat2(rot.x) * *self }

  #[inline(always)]
  fn rotate(&mut self, rot: &Vec1<N>)
  { *self = self.rotated(rot) }
}

impl<N: Div<N, N> + Trigonometric + Neg<N> + Mul<N, N> + Add<N, N> + Copy +
        One + Sub<N, N>>
Rotation<(Vec3<N>, N)> for Rotmat<Mat3<N>>
{
  #[inline(always)]
  fn rotation(&self) -> (Vec3<N>, N)
  { fail!("Not yet implemented.") }

  #[inline(always)]
  fn rotated(&self, &(axis, angle): &(Vec3<N>, N)) -> Rotmat<Mat3<N>>
  { rotmat3(&axis, angle) * *self }

  #[inline(always)]
  fn rotate(&mut self, rot: &(Vec3<N>, N))
  { *self = self.rotated(rot) }
}

impl<N: Copy + Rand + Trigonometric + Neg<N>> Rand for Rotmat<Mat2<N>>
{
  #[inline(always)]
  fn rand<R: Rng>(rng: &mut R) -> Rotmat<Mat2<N>>
  { rotmat2(rng.gen()) }
}

impl<N: Copy + Rand + Trigonometric + Neg<N> + One + Sub<N, N> + Add<N, N> +
       Mul<N, N>>
Rand for Rotmat<Mat3<N>>
{
  #[inline(always)]
  fn rand<R: Rng>(rng: &mut R) -> Rotmat<Mat3<N>>
  { rotmat3(&rng.gen(), rng.gen()) }
}

impl<M: Dim> Dim for Rotmat<M>
{
  #[inline(always)]
  fn dim() -> uint
  { Dim::dim::<M>() }
}
 
impl<M: Copy + One + Zero> One for Rotmat<M>
{
  #[inline(always)]
  fn one() -> Rotmat<M>
  { Rotmat { submat: One::one() } }
}

impl<M: Copy + Mul<M, M>> Mul<Rotmat<M>, Rotmat<M>> for Rotmat<M>
{
  #[inline(always)]
  fn mul(&self, other: &Rotmat<M>) -> Rotmat<M>
  { Rotmat { submat: self.submat.mul(&other.submat) } }
}

impl<V, M: RMul<V>> RMul<V> for Rotmat<M>
{
  #[inline(always)]
  fn rmul(&self, other: &V) -> V
  { self.submat.rmul(other) }
}

impl<V, M: LMul<V>> LMul<V> for Rotmat<M>
{
  #[inline(always)]
  fn lmul(&self, other: &V) -> V
  { self.submat.lmul(other) }
}

impl<M: Copy> DeltaTransform<M> for Rotmat<M>
{
  #[inline(always)]
  fn delta_transform(&self) -> M
  { self.submat }
}

impl<M: RMul<V> + Copy, V: Copy> DeltaTransformVector<V> for Rotmat<M>
{
  #[inline(always)]
  fn delta_transform_vector(&self, v: &V) -> V
  { self.submat.rmul(v) }
}

impl<M: Copy + Transpose> Inv for Rotmat<M>
{
  #[inline(always)]
  fn invert(&mut self)
  { self.transpose() }

  #[inline(always)]
  fn inverse(&self) -> Rotmat<M>
  { self.transposed() }
}

impl<M: Copy + Transpose>
Transpose for Rotmat<M>
{
  #[inline(always)]
  fn transposed(&self) -> Rotmat<M>
  { Rotmat { submat: self.submat.transposed() } }

  #[inline(always)]
  fn transpose(&mut self)
  { self.submat.transpose() }
}

impl<N: ApproxEq<N>, M: ApproxEq<N>> ApproxEq<N> for Rotmat<M>
{
  #[inline(always)]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline(always)]
  fn approx_eq(&self, other: &Rotmat<M>) -> bool
  { self.submat.approx_eq(&other.submat) }

  #[inline(always)]
  fn approx_eq_eps(&self, other: &Rotmat<M>, epsilon: &N) -> bool
  { self.submat.approx_eq_eps(&other.submat, epsilon) }
}
