use std::num::{Zero, One};
use mat::{Mat1, Mat2, Mat3};
use traits::division_ring::DivisionRing;
use traits::inv::Inv;

// some specializations:
impl<N: DivisionRing + Clone>
Inv for Mat1<N> {
  #[inline]
  fn inverse(&self) -> Option<Mat1<N>> {
    let mut res : Mat1<N> = self.clone();

    if res.inplace_inverse() {
        Some(res)
    }
    else {
        None
    }
  }

  #[inline]
  fn inplace_inverse(&mut self) -> bool {
    if self.m11.is_zero() {
        false
    }
    else {
      self.m11 = One::one::<N>() / self.m11;
      true
    }
  }
}

impl<N: DivisionRing + Clone>
Inv for Mat2<N> {
  #[inline]
  fn inverse(&self) -> Option<Mat2<N>> {
    let mut res : Mat2<N> = self.clone();

    if res.inplace_inverse() {
        Some(res)
    }
    else {
        None
    }
  }

  #[inline]
  fn inplace_inverse(&mut self) -> bool {
    let det = self.m11 * self.m22 - self.m21 * self.m12;

    if det.is_zero() {
        false
    }
    else {
      *self = Mat2::new(self.m22 / det , -self.m12 / det,
                        -self.m21 / det, self.m11 / det);

      true
    }
  }
}

impl<N: DivisionRing + Clone>
Inv for Mat3<N> {
  #[inline]
  fn inverse(&self) -> Option<Mat3<N>> {
    let mut res = self.clone();

    if res.inplace_inverse() {
        Some(res)
    }
    else {
        None
    }
  }

  #[inline]
  fn inplace_inverse(&mut self) -> bool {
    let minor_m12_m23 = self.m22 * self.m33 - self.m32 * self.m23;
    let minor_m11_m23 = self.m21 * self.m33 - self.m31 * self.m23;
    let minor_m11_m22 = self.m21 * self.m32 - self.m31 * self.m22;

    let det = self.m11 * minor_m12_m23
              - self.m12 * minor_m11_m23
              + self.m13 * minor_m11_m22;

    if det.is_zero() {
        false
    }
    else {
      *self = Mat3::new(
        (minor_m12_m23  / det),
        ((self.m13 * self.m32 - self.m33 * self.m12) / det),
        ((self.m12 * self.m23 - self.m22 * self.m13) / det),

        (-minor_m11_m23 / det),
        ((self.m11 * self.m33 - self.m31 * self.m13) / det),
        ((self.m13 * self.m21 - self.m23 * self.m11) / det),

        (minor_m11_m22  / det),
        ((self.m12 * self.m31 - self.m32 * self.m11) / det),
        ((self.m11 * self.m22 - self.m21 * self.m12) / det)
      );

      true
    }
  }
}
