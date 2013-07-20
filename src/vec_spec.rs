use std::num::{Zero, One};
use traits::basis::Basis;
use traits::cross::Cross;
use traits::division_ring::DivisionRing;
use traits::norm::Norm;
use traits::sample::UniformSphereSample;
use vec::{Vec1, Vec2, Vec3};

// FIXME: impl<N, Iter: Iterator<N>> FromIterator<N, Iter> for Vec0<N>
// FIXME: {
// FIXME:   fn from_iterator(_: &mut Iter) -> Vec0<N>
// FIXME:   { Vec0 { at: [ ] } }
// FIXME: }

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec1<N>> for Vec2<N>
{
  #[inline]
  fn cross(&self, other : &Vec2<N>) -> Vec1<N>
  { Vec1::new(self.x * other.y - self.y * other.x) }
}

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec3<N>> for Vec3<N>
{
  #[inline]
  fn cross(&self, other : &Vec3<N>) -> Vec3<N>
  {
    Vec3::new(self.y * other.z - self.z * other.y,
              self.z * other.x - self.x * other.z,
              self.x * other.y - self.y * other.x
    )
  }
}

impl<N: One> Basis for Vec1<N>
{
  #[inline(always)]
  fn canonical_basis(f: &fn(Vec1<N>))
  { f(Vec1::new(One::one())) }

  #[inline(always)]
  fn orthonormal_subspace_basis(&self, _: &fn(Vec1<N>))
  { }
}

impl<N: Clone + One + Zero + Neg<N>> Basis for Vec2<N>
{
  #[inline]
  fn canonical_basis(f: &fn(Vec2<N>))
  {
    f(Vec2::new(One::one(), Zero::zero()));
    f(Vec2::new(Zero::zero(), One::one()));
  }

  #[inline]
  fn orthonormal_subspace_basis(&self, f: &fn(Vec2<N>))
  { f(Vec2::new(-self.y, self.x.clone())) }
}

impl<N: Clone + DivisionRing + Ord + Algebraic + Signed>
Basis for Vec3<N>
{
  #[inline(always)]
  fn canonical_basis(f: &fn(Vec3<N>))
  {
    f(Vec3::new(One::one(), Zero::zero(), Zero::zero()));
    f(Vec3::new(Zero::zero(), One::one(), Zero::zero()));
    f(Vec3::new(Zero::zero(), Zero::zero(), One::one()));
  }

  #[inline(always)]
  fn orthonormal_subspace_basis(&self, f: &fn(Vec3<N>))
  {
      let a = 
        if self.x.clone().abs() > self.y.clone().abs()
        { Vec3::new(self.z.clone(), Zero::zero(), -self.x).normalized() }
        else
        { Vec3::new(Zero::zero(), -self.z, self.y.clone()).normalized() };

      f(a.cross(self));
      f(a);
  }
}

// FIXME: this bad: this fixes definitly the number of samples…
static SAMPLES_2_F64: [Vec2<f64>, ..21] = [
  Vec2 { x: 1.0,         y: 0.0         },
  Vec2 { x: 0.95557281,  y: 0.29475517  },
  Vec2 { x: 0.82623877,  y: 0.56332006  },
  Vec2 { x: 0.6234898,   y: 0.78183148  },
  Vec2 { x: 0.36534102,  y: 0.93087375  },
  Vec2 { x: 0.07473009,  y: 0.9972038   },
  Vec2 { x: -0.22252093, y: 0.97492791  },
  Vec2 { x: -0.5,        y: 0.8660254   },
  Vec2 { x: -0.73305187, y: 0.68017274  },
  Vec2 { x: -0.90096887, y: 0.43388374  },
  Vec2 { x: -0.98883083, y: 0.14904227  },
  Vec2 { x: -0.98883083, y: -0.14904227 },
  Vec2 { x: -0.90096887, y: -0.43388374 },
  Vec2 { x: -0.73305187, y: -0.68017274 },
  Vec2 { x: -0.5,        y: -0.8660254  },
  Vec2 { x: -0.22252093, y: -0.97492791 },
  Vec2 { x: 0.07473009,  y: -0.9972038  },
  Vec2 { x: 0.36534102,  y: -0.93087375 },
  Vec2 { x: 0.6234898,   y: -0.78183148 },
  Vec2 { x: 0.82623877,  y: -0.56332006 },
  Vec2 { x: 0.95557281,  y: -0.29475517 },
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
