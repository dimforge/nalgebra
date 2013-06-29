use std::num::{Zero, One};
use mat::{Mat1, Mat2, Mat3};
use traits::division_ring::DivisionRing;
use traits::inv::Inv;

// some specializations:
impl<N: Copy + DivisionRing>
Inv for Mat1<N>
{
  #[inline]
  fn inverse(&self) -> Mat1<N>
  {
    let mut res : Mat1<N> = copy *self;

    res.invert();

    res
  }

  #[inline]
  fn invert(&mut self)
  {
    assert!(!self.mij[0].is_zero());

    self.mij[0] = One::one::<N>() / self.mij[0]
  }
}

impl<N: Copy + DivisionRing>
Inv for Mat2<N>
{
  #[inline]
  fn inverse(&self) -> Mat2<N>
  {
    let mut res : Mat2<N> = copy *self;

    res.invert();

    res
  }

  #[inline]
  fn invert(&mut self)
  {
    let det = self.mij[0 * 2 + 0] * self.mij[1 * 2 + 1] - self.mij[1 * 2 + 0] * self.mij[0 * 2 + 1];

    assert!(!det.is_zero());

    *self = Mat2::new([self.mij[1 * 2 + 1] / det , -self.mij[0 * 2 + 1] / det,
                           -self.mij[1 * 2 + 0] / det, self.mij[0 * 2 + 0] / det])
  }
}

impl<N: Copy + DivisionRing>
Inv for Mat3<N>
{
  #[inline]
  fn inverse(&self) -> Mat3<N>
  {
    let mut res = copy *self;

    res.invert();

    res
  }

  #[inline]
  fn invert(&mut self)
  {
    let minor_m12_m23 = self.mij[1 * 3 + 1] * self.mij[2 * 3 + 2] - self.mij[2 * 3 + 1] * self.mij[1 * 3 + 2];
    let minor_m11_m23 = self.mij[1 * 3 + 0] * self.mij[2 * 3 + 2] - self.mij[2 * 3 + 0] * self.mij[1 * 3 + 2];
    let minor_m11_m22 = self.mij[1 * 3 + 0] * self.mij[2 * 3 + 1] - self.mij[2 * 3 + 0] * self.mij[1 * 3 + 1];

    let det = self.mij[0 * 3 + 0] * minor_m12_m23
              - self.mij[0 * 3 + 1] * minor_m11_m23
              + self.mij[0 * 3 + 2] * minor_m11_m22;

    assert!(!det.is_zero());

    *self = Mat3::new( [
      (minor_m12_m23  / det),
      ((self.mij[0 * 3 + 2] * self.mij[2 * 3 + 1] - self.mij[2 * 3 + 2] * self.mij[0 * 3 + 1]) / det),
      ((self.mij[0 * 3 + 1] * self.mij[1 * 3 + 2] - self.mij[1 * 3 + 1] * self.mij[0 * 3 + 2]) / det),

      (-minor_m11_m23 / det),
      ((self.mij[0 * 3 + 0] * self.mij[2 * 3 + 2] - self.mij[2 * 3 + 0] * self.mij[0 * 3 + 2]) / det),
      ((self.mij[0 * 3 + 2] * self.mij[1 * 3 + 0] - self.mij[1 * 3 + 2] * self.mij[0 * 3 + 0]) / det),

      (minor_m11_m22  / det),
      ((self.mij[0 * 3 + 1] * self.mij[2 * 3 + 0] - self.mij[2 * 3 + 1] * self.mij[0 * 3 + 0]) / det),
      ((self.mij[0 * 3 + 0] * self.mij[1 * 3 + 1] - self.mij[1 * 3 + 0] * self.mij[0 * 3 + 1]) / det)
    ] )
  }
}
