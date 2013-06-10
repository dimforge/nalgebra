use std::num::{Zero, One, Algebraic, abs};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::basis::Basis;
use traits::cross::Cross;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::flatten::Flatten;
use traits::translation::Translation;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, ToStr)]
pub struct Vec3<N>
{
  x : N,
  y : N,
  z : N
}

impl<N: Copy> Vec3<N>
{
  pub fn new(x: N, y: N, z: N) -> Vec3<N>
  { Vec3 {x: x, y: y, z: z} }
}

impl<N> Dim for Vec3<N>
{
  fn dim() -> uint
  { 3 }
}

impl<N:Copy + Add<N,N>> Add<Vec3<N>, Vec3<N>> for Vec3<N>
{
  fn add(&self, other: &Vec3<N>) -> Vec3<N>
  { Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z) }
}

impl<N:Copy + Sub<N,N>> Sub<Vec3<N>, Vec3<N>> for Vec3<N>
{
  fn sub(&self, other: &Vec3<N>) -> Vec3<N>
  { Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z) }
}

impl<N: Copy + Mul<N, N>>
ScalarMul<N> for Vec3<N>
{
  fn scalar_mul(&self, s: &N) -> Vec3<N>
  { Vec3 { x: self.x * *s, y: self.y * *s, z: self.z * *s } }

  fn scalar_mul_inplace(&mut self, s: &N)
  {
    self.x *= *s;
    self.y *= *s;
    self.z *= *s;
  }
}


impl<N: Copy + Div<N, N>>
ScalarDiv<N> for Vec3<N>
{
  fn scalar_div(&self, s: &N) -> Vec3<N>
  { Vec3 { x: self.x / *s, y: self.y / *s, z: self.z / *s } }

  fn scalar_div_inplace(&mut self, s: &N)
  {
    self.x /= *s;
    self.y /= *s;
    self.z /= *s;
  }
}

impl<N: Copy + Add<N, N>>
ScalarAdd<N> for Vec3<N>
{
  fn scalar_add(&self, s: &N) -> Vec3<N>
  { Vec3 { x: self.x + *s, y: self.y + *s, z: self.z + *s } }

  fn scalar_add_inplace(&mut self, s: &N)
  {
    self.x += *s;
    self.y += *s;
    self.z += *s;
  }
}

impl<N: Copy + Sub<N, N>>
ScalarSub<N> for Vec3<N>
{
  fn scalar_sub(&self, s: &N) -> Vec3<N>
  { Vec3 { x: self.x - *s, y: self.y - *s, z: self.z - *s } }

  fn scalar_sub_inplace(&mut self, s: &N)
  {
    self.x -= *s;
    self.y -= *s;
    self.z -= *s;
  }
}

impl<N: Copy + Add<N, N>> Translation<Vec3<N>> for Vec3<N>
{
  fn translation(&self) -> Vec3<N>
  { *self }

  fn translated(&self, t: &Vec3<N>) -> Vec3<N>
  { self + *t }

  fn translate(&mut self, t: &Vec3<N>)
  { *self += *t; }
}



impl<N:Copy + Neg<N>> Neg<Vec3<N>> for Vec3<N>
{
  fn neg(&self) -> Vec3<N>
  { Vec3::new(-self.x, -self.y, -self.z) }
}

impl<N:Copy + Mul<N, N> + Add<N, N>> Dot<N> for Vec3<N>
{
  fn dot(&self, other : &Vec3<N>) -> N
  { self.x * other.x + self.y * other.y + self.z * other.z } 
}

impl<N:Copy + Mul<N, N> + Add<N, N> + Sub<N, N>> SubDot<N> for Vec3<N>
{
  fn sub_dot(&self, a: &Vec3<N>, b: &Vec3<N>) -> N
  { (self.x - a.x) * b.x + (self.y - a.y) * b.y + (self.z - a.z) * b.z } 
}

impl<N:Copy + Mul<N, N> + Add<N, N> + Div<N, N> + Algebraic>
Norm<N> for Vec3<N>
{
  fn sqnorm(&self) -> N
  { self.dot(self) }

  fn norm(&self) -> N
  { self.sqnorm().sqrt() }

  fn normalized(&self) -> Vec3<N>
  {
    let l = self.norm();

    Vec3::new(self.x / l, self.y / l, self.z / l)
  }

  fn normalize(&mut self) -> N
  {
    let l = self.norm();

    self.x /= l;
    self.y /= l;
    self.z /= l;

    l
  }
}

impl<N:Copy + Mul<N, N> + Sub<N, N>> Cross<Vec3<N>> for Vec3<N>
{
  fn cross(&self, other : &Vec3<N>) -> Vec3<N>
  {
    Vec3::new(
      self.y * other.z - self.z * other.y,
      self.z * other.x - self.x * other.z,
      self.x * other.y - self.y * other.x
    )
  }
}

impl<N:Copy + Zero> Zero for Vec3<N>
{
  fn zero() -> Vec3<N>
  {
    let _0 = Zero::zero();
    Vec3::new(_0, _0, _0)
  }

  fn is_zero(&self) -> bool
  { self.x.is_zero() && self.y.is_zero() && self.z.is_zero() }
}

impl<N: Copy + One + Zero + Neg<N> + Ord + Mul<N, N> + Sub<N, N> + Add<N, N> +
        Div<N, N> + Algebraic>
Basis for Vec3<N>
{
  fn canonical_basis() -> ~[Vec3<N>]
  {
    // FIXME: this should be static
    ~[ Vec3::new(One::one(), Zero::zero(), Zero::zero()),
       Vec3::new(Zero::zero(), One::one(), Zero::zero()),
       Vec3::new(Zero::zero(), Zero::zero(), One::one()) ]
  }

  fn orthogonal_subspace_basis(&self) -> ~[Vec3<N>]
  {
      let a = 
        if (abs(self.x) > abs(self.y))
        { Vec3::new(self.z, Zero::zero(), -self.x).normalized() }
        else
        { Vec3::new(Zero::zero(), -self.z, self.y).normalized() };

      ~[ a, a.cross(self) ]
  }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Vec3<N>
{
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  fn approx_eq(&self, other: &Vec3<N>) -> bool
  {
    self.x.approx_eq(&other.x) &&
    self.y.approx_eq(&other.y) &&
    self.z.approx_eq(&other.z)
  }

  fn approx_eq_eps(&self, other: &Vec3<N>, epsilon: &N) -> bool
  {
    self.x.approx_eq_eps(&other.x, epsilon) &&
    self.y.approx_eq_eps(&other.y, epsilon) &&
    self.z.approx_eq_eps(&other.z, epsilon)
  }
}

impl<N:Copy + Rand> Rand for Vec3<N>
{
  fn rand<R: Rng>(rng: &mut R) -> Vec3<N>
  { Vec3::new(rng.gen(), rng.gen(), rng.gen()) }
}

impl<N: Copy> Flatten<N> for Vec3<N>
{
  fn flat_size() -> uint
  { 3 }

  fn from_flattened(l: &[N], off: uint) -> Vec3<N>
  { Vec3::new(l[off], l[off + 1], l[off + 2]) }

  fn flatten(&self) -> ~[N]
  { ~[ self.x, self.y, self.z ] }

  fn flatten_to(&self, l: &mut [N], off: uint)
  {
    l[off]     = self.x;
    l[off + 1] = self.y;
    l[off + 2] = self.z;
  }
}
