use std::vec::{VecIterator, VecMutIterator};
use std::iterator::FromIterator;
use std::num::{Zero, One};
use traits::basis::Basis;
use traits::cross::Cross;
use traits::division_ring::DivisionRing;
use traits::norm::Norm;
use traits::sample::UniformSphereSample;
use traits::iterable::FromAnyIterator;
use vec::{Vec0, Vec1, Vec2, Vec3};

impl<N: Clone> FromAnyIterator<N> for Vec0<N>
{
  fn from_iterator<'l>(_: &mut VecIterator<'l, N>) -> Vec0<N>
  { Vec0 { at: [ ] } }

  fn from_mut_iterator<'l>(_: &mut VecMutIterator<'l, N>) -> Vec0<N>
  { Vec0 { at: [ ] } }
}

impl<N, Iter: Iterator<N>> FromIterator<N, Iter> for Vec0<N>
{
  fn from_iterator(_: &mut Iter) -> Vec0<N>
  { Vec0 { at: [ ] } }
}

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
  #[inline(always)]
  fn canonical_basis(f: &fn(Vec1<N>))
  { f(Vec1::new([One::one()])) }

  #[inline(always)]
  fn orthonormal_subspace_basis(&self, _: &fn(Vec1<N>))
  { }
}

impl<N: Clone + One + Zero + Neg<N>> Basis for Vec2<N>
{
  #[inline]
  fn canonical_basis(f: &fn(Vec2<N>))
  {
    f(Vec2::new([One::one(), Zero::zero()]));
    f(Vec2::new([Zero::zero(), One::one()]));
  }

  #[inline]
  fn orthonormal_subspace_basis(&self, f: &fn(Vec2<N>))
  { f(Vec2::new([-self.at[1], self.at[0].clone()])) }
}

impl<N: Clone + Copy + DivisionRing + Ord + Algebraic + Signed>
Basis for Vec3<N>
{
  #[inline(always)]
  fn canonical_basis(f: &fn(Vec3<N>))
  {
    f(Vec3::new([One::one(), Zero::zero(), Zero::zero()]));
    f(Vec3::new([Zero::zero(), One::one(), Zero::zero()]));
    f(Vec3::new([Zero::zero(), Zero::zero(), One::one()]));
  }

  #[inline(always)]
  fn orthonormal_subspace_basis(&self, f: &fn(Vec3<N>))
  {
      let a = 
        if self.at[0].clone().abs() > self.at[1].clone().abs()
        { Vec3::new([self.at[2].clone(), Zero::zero(), -self.at[0]]).normalized() }
        else
        { Vec3::new([Zero::zero(), -self.at[2], self.at[1].clone()]).normalized() };

      f(a.cross(self));
      f(a);
  }
}

// FIXME: this bad: this fixes definitly the number of samples…
static SAMPLES_2_F64: [Vec2<f64>, ..21] = [
  Vec2 { at: [1.0, 0.0] },
  Vec2 { at: [0.95557281, 0.29475517] },
  Vec2 { at: [0.82623877, 0.56332006] },
  Vec2 { at: [0.6234898, 0.78183148] },
  Vec2 { at: [0.36534102, 0.93087375] },
  Vec2 { at: [0.07473009, 0.9972038] },
  Vec2 { at: [-0.22252093, 0.97492791] },
  Vec2 { at: [-0.5, 0.8660254] },
  Vec2 { at: [-0.73305187, 0.68017274] },
  Vec2 { at: [-0.90096887, 0.43388374] },
  Vec2 { at: [-0.98883083, 0.14904227] },
  Vec2 { at: [-0.98883083, -0.14904227] },
  Vec2 { at: [-0.90096887, -0.43388374] },
  Vec2 { at: [-0.73305187, -0.68017274] },
  Vec2 { at: [-0.5, -0.8660254] },
  Vec2 { at: [-0.22252093, -0.97492791] },
  Vec2 { at: [0.07473009, -0.9972038] },
  Vec2 { at: [0.36534102, -0.93087375] },
  Vec2 { at: [0.6234898, -0.78183148] },
  Vec2 { at: [0.82623877, -0.56332006] },
  Vec2 { at: [0.95557281, -0.29475517] },
];

impl UniformSphereSample for Vec2<f64>
{
  pub fn sample(f: &fn(&'static Vec2<f64>))
  {
    for SAMPLES_2_F64.iter().advance |sample|
    { f(sample) }
  }
}

impl UniformSphereSample for Vec3<f64>
{
  pub fn sample(_: &fn(&'static Vec3<f64>))
  {
    fail!("UniformSphereSample for Vec3<f64> is not yet implemented.")
  }
}
