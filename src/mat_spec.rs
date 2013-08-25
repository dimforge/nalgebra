use std::num::{Zero, One};
use vec::Vec3;
use mat::{Mat1, Mat2, Mat3};
use traits::inv::Inv;
use traits::row::Row;

// some specializations:
impl<N: Num + Clone>
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

impl<N: Num + Clone>
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
            *self = Mat2::new(
                self.m22 / det , -self.m12 / det,
                -self.m21 / det, self.m11 / det);

            true
        }
    }
}

impl<N: Num + Clone>
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
                (minor_m12_m23 / det),
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

impl<N: Clone> Row<Vec3<N>> for Mat3<N> {
    #[inline]
    fn row(&self, i: uint) -> Vec3<N> {
        match i {
            0 => Vec3::new(self.m11.clone(), self.m12.clone(), self.m13.clone()),
            1 => Vec3::new(self.m21.clone(), self.m22.clone(), self.m23.clone()),
            2 => Vec3::new(self.m31.clone(), self.m32.clone(), self.m33.clone()),
            _ => fail!("Index out of range: 3d matrices do not have " + i.to_str() + " rows.")
        }
    }

    #[inline]
    fn set_row(&mut self, i: uint, r: Vec3<N>) {
        match i {
            0 => {
                self.m11 = r.x.clone();
                self.m12 = r.y.clone();
                self.m13 = r.z;
            },
            1 => {
                self.m21 = r.x.clone();
                self.m22 = r.y.clone();
                self.m23 = r.z;
            },
            2 => {
                self.m31 = r.x.clone();
                self.m32 = r.y.clone();
                self.m33 = r.z;
            },
            _ => fail!("Index out of range: 3d matrices do not have " + i.to_str() + " rows.")

        }
    }
}
