use std::num::{Zero, One, abs};
use traits::basis::Basis;
use traits::cross::Cross;
use traits::division_ring::DivisionRing;
use traits::norm::Norm;
use vec::{Vec1, Vec2, Vec3};

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec1<N>> for Vec2<N>
{
  #[inline]
  fn cross(&self, other : &Vec2<N>) -> Vec1<N>
  { Vec1::new([self.at[0] * other.at[1] - self.at[1] * other.at[0]]) }
}

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec3<N>> for Vec3<N>
{
  #[inline]
  fn cross(&self, other : &Vec3<N>) -> Vec3<N>
  {
    Vec3::new(
      [self.at[1] * other.at[2] - self.at[2] * other.at[1],
       self.at[2] * other.at[0] - self.at[0] * other.at[2],
       self.at[0] * other.at[1] - self.at[1] * other.at[0]]
    )
  }
}

impl<N: One> Basis for Vec1<N>
{
  #[inline]
  fn canonical_basis() -> ~[Vec1<N>]
  { ~[ Vec1::new([One::one()]) ] } // FIXME: this should be static

  #[inline]
  fn orthogonal_subspace_basis(&self) -> ~[Vec1<N>]
  { ~[] }
}

impl<N: Copy + One + Zero + Neg<N>> Basis for Vec2<N>
{
  #[inline]
  fn canonical_basis()     -> ~[Vec2<N>]
  {
    // FIXME: this should be static
    ~[ Vec2::new([One::one(), Zero::zero()]),
       Vec2::new([Zero::zero(), One::one()]) ]
  }

  #[inline]
  fn orthogonal_subspace_basis(&self) -> ~[Vec2<N>]
  { ~[ Vec2::new([-self.at[1], copy self.at[0]]) ] }
}

impl<N: Copy + DivisionRing + Ord + Algebraic>
Basis for Vec3<N>
{
  #[inline]
  fn canonical_basis() -> ~[Vec3<N>]
  {
    // FIXME: this should be static
    ~[ Vec3::new([One::one(), Zero::zero(), Zero::zero()]),
       Vec3::new([Zero::zero(), One::one(), Zero::zero()]),
       Vec3::new([Zero::zero(), Zero::zero(), One::one()]) ]
  }

  #[inline]
  fn orthogonal_subspace_basis(&self) -> ~[Vec3<N>]
  {
      let a = 
        if abs(copy self.at[0]) > abs(copy self.at[1])
        { Vec3::new([copy self.at[2], Zero::zero(), -copy self.at[0]]).normalized() }
        else
        { Vec3::new([Zero::zero(), -self.at[2], copy self.at[1]]).normalized() };

      ~[ a.cross(self), a ]
  }
}
