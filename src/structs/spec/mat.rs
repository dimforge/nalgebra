use std::num::{Zero, One};
use structs::vec::{Vec2, Vec3, Vec2MulRhs, Vec3MulRhs};
use structs::mat::{Mat1, Mat2, Mat3, Mat3MulRhs, Mat2MulRhs};
use structs::mat;
use traits::operations::{Inv};
use traits::structure::{Row, Col};

// some specializations:
impl<N: Num + Clone>
Inv for Mat1<N> {
    #[inline]
    fn inverted(&self) -> Option<Mat1<N>> {
        let mut res : Mat1<N> = self.clone();

        if res.invert() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn invert(&mut self) -> bool {
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
    fn inverted(&self) -> Option<Mat2<N>> {
        let mut res : Mat2<N> = self.clone();

        if res.invert() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn invert(&mut self) -> bool {
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
    fn inverted(&self) -> Option<Mat3<N>> {
        let mut res = self.clone();

        if res.invert() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn invert(&mut self) -> bool {
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
    fn nrows(&self) -> uint {
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
    fn ncols(&self) -> uint {
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
    fn binop(left: &Mat3<N>, right: &Mat3<N>) -> Mat3<N> {
        Mat3::new(
            left.m11 * right.m11 + left.m12 * right.m21 + left.m13 * right.m31,
            left.m11 * right.m12 + left.m12 * right.m22 + left.m13 * right.m32,
            left.m11 * right.m13 + left.m12 * right.m23 + left.m13 * right.m33,

            left.m21 * right.m11 + left.m22 * right.m21 + left.m23 * right.m31,
            left.m21 * right.m12 + left.m22 * right.m22 + left.m23 * right.m32,
            left.m21 * right.m13 + left.m22 * right.m23 + left.m23 * right.m33,

            left.m31 * right.m11 + left.m32 * right.m21 + left.m33 * right.m31,
            left.m31 * right.m12 + left.m32 * right.m22 + left.m33 * right.m32,
            left.m31 * right.m13 + left.m32 * right.m23 + left.m33 * right.m33
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> Mat2MulRhs<N, Mat2<N>> for Mat2<N> {
    #[inline(always)]
    fn binop(left: &Mat2<N>, right: &Mat2<N>) -> Mat2<N> {
        Mat2::new(
            left.m11 * right.m11 + left.m12 * right.m21,
            left.m11 * right.m12 + left.m12 * right.m22,

            left.m21 * right.m11 + left.m22 * right.m21,
            left.m21 * right.m12 + left.m22 * right.m22
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> Mat3MulRhs<N, Vec3<N>> for Vec3<N> {
    #[inline(always)]
    fn binop(left: &Mat3<N>, right: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            left.m11 * right.x + left.m12 * right.y + left.m13 * right.z,
            left.m21 * right.x + left.m22 * right.y + left.m23 * right.z,
            left.m31 * right.x + left.m32 * right.y + left.m33 * right.z
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> Vec3MulRhs<N, Vec3<N>> for Mat3<N> {
    #[inline(always)]
    fn binop(left: &Vec3<N>, right: &Mat3<N>) -> Vec3<N> {
        Vec3::new(
            left.x * right.m11 + left.y * right.m21 + left.z * right.m31,
            left.x * right.m12 + left.y * right.m22 + left.z * right.m32,
            left.x * right.m13 + left.y * right.m23 + left.z * right.m33
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> Vec2MulRhs<N, Vec2<N>> for Mat2<N> {
    #[inline(always)]
    fn binop(left: &Vec2<N>, right: &Mat2<N>) -> Vec2<N> {
        Vec2::new(
            left.x * right.m11 + left.y * right.m21,
            left.x * right.m12 + left.y * right.m22
        )
    }
}

impl<N: Mul<N, N> + Add<N, N>> Mat2MulRhs<N, Vec2<N>> for Vec2<N> {
    #[inline(always)]
    fn binop(left: &Mat2<N>, right: &Vec2<N>) -> Vec2<N> {
        Vec2::new(
            left.m11 * right.x + left.m12 * right.y,
            left.m21 * right.x + left.m22 * right.y
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
