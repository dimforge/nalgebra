use std::num::{Zero, One};
use vec::{Vec2, Vec3};
use mat::{Mat1, Mat2, Mat3, Inv, Row, Col, RMul, LMul, Mat3MulRhs, Mat2MulRhs};
use mat;

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
            let _1: N = One::one();
            self.m11 = _1 / self.m11;
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
    fn num_rows(&self) -> uint {
        3
    }

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

impl<N: Clone> Col<Vec3<N>> for Mat3<N> {
    #[inline]
    fn num_cols(&self) -> uint {
        3
    }

    #[inline]
    fn col(&self, i: uint) -> Vec3<N> {
        match i {
            0 => Vec3::new(self.m11.clone(), self.m21.clone(), self.m31.clone()),
            1 => Vec3::new(self.m12.clone(), self.m22.clone(), self.m32.clone()),
            2 => Vec3::new(self.m13.clone(), self.m23.clone(), self.m33.clone()),
            _ => fail!("Index out of range: 3d matrices do not have " + i.to_str() + " cols.")
        }
    }

    #[inline]
    fn set_col(&mut self, i: uint, r: Vec3<N>) {
        match i {
            0 => {
                self.m11 = r.x.clone();
                self.m21 = r.y.clone();
                self.m31 = r.z;
            },
            1 => {
                self.m12 = r.x.clone();
                self.m22 = r.y.clone();
                self.m32 = r.z;
            },
            2 => {
                self.m13 = r.x.clone();
                self.m23 = r.y.clone();
                self.m33 = r.z;
            },
            _ => fail!("Index out of range: 3d matrices do not have " + i.to_str() + " cols.")

        }
    }
}

impl<N: Mul<N, N> + Add<N, N>> Mat3MulRhs<N, Mat3<N>> for Mat3<N> {
    #[inline]
    fn Mat3MulRhs(&self, other: &Mat3<N>) -> Mat3<N> {
        Mat3::new(
            other.m11 * self.m11 + other.m12 * self.m21 + other.m13 * self.m31,
            other.m11 * self.m12 + other.m12 * self.m22 + other.m13 * self.m32,
            other.m11 * self.m13 + other.m12 * self.m23 + other.m13 * self.m33,

            other.m21 * self.m11 + other.m22 * self.m21 + other.m23 * self.m31,
            other.m21 * self.m12 + other.m22 * self.m22 + other.m23 * self.m32,
            other.m21 * self.m13 + other.m22 * self.m23 + other.m23 * self.m33,

            other.m31 * self.m11 + other.m32 * self.m21 + other.m33 * self.m31,
            other.m31 * self.m12 + other.m32 * self.m22 + other.m33 * self.m32,
            other.m31 * self.m13 + other.m32 * self.m23 + other.m33 * self.m33
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> Mat2MulRhs<N, Mat2<N>> for Mat2<N> {
    #[inline(always)]
    fn Mat2MulRhs(&self, other: &Mat2<N>) -> Mat2<N> {
        Mat2::new(
            other.m11 * self.m11 + other.m12 * self.m21,
            other.m11 * self.m12 + other.m12 * self.m22,

            other.m21 * self.m11 + other.m22 * self.m21,
            other.m21 * self.m12 + other.m22 * self.m22
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> RMul<Vec3<N>> for Mat3<N> {
    #[inline(always)]
    fn rmul(&self, v: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            self.m11 * v.x + self.m12 * v.y + self.m13 * v.z,
            self.m21 * v.x + self.m22 * v.y + self.m23 * v.z,
            self.m31 * v.x + self.m32 * v.y + self.m33 * v.z
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> LMul<Vec3<N>> for Mat3<N> {
    #[inline(always)]
    fn lmul(&self, v: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            self.m11 * v.x + self.m21 * v.y + self.m31 * v.z,
            self.m12 * v.x + self.m22 * v.y + self.m32 * v.z,
            self.m13 * v.x + self.m23 * v.y + self.m33 * v.z
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> RMul<Vec2<N>> for Mat2<N> {
    #[inline(always)]
    fn rmul(&self, v: &Vec2<N>) -> Vec2<N> {
        Vec2::new(
            self.m11 * v.x + self.m12 * v.y,
            self.m21 * v.x + self.m22 * v.y
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> LMul<Vec2<N>> for Mat2<N> {
    #[inline(always)]
    fn lmul(&self, v: &Vec2<N>) -> Vec2<N> {
        Vec2::new(
            self.m11 * v.x + self.m21 * v.y,
            self.m12 * v.x + self.m22 * v.y
        )
    }
}

// FIXME: move this to another file?
impl<N: Real + NumCast + Zero + One> mat::Mat4<N> {
    /// Computes a projection matrix given the frustrum near plane width, height, the field of
    /// view, and the distance to the clipping planes (`znear` and `zfar`).
    pub fn new_perspective(width: N, height: N, fov: N, znear: N, zfar: N) -> mat::Mat4<N> {
        let aspect = width / height;

        let _1: N = One::one();
        let sy    = _1 / (fov * NumCast::from(0.5)).tan();
        let sx    = -sy / aspect;
        let sz    = -(zfar + znear) / (znear - zfar);
        let tz    = zfar * znear * NumCast::from(2.0) / (znear - zfar);

        mat::Mat4::new(
            sx,           Zero::zero(), Zero::zero(), Zero::zero(),
            Zero::zero(), sy,           Zero::zero(), Zero::zero(),
            Zero::zero(), Zero::zero(), sz,           tz,
            Zero::zero(), Zero::zero(), One::one(),   Zero::zero()
            )
    }
}
