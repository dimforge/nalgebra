use structs::vec::{Vec2, Vec3};
use structs::pnt::{Pnt2, Pnt3};
use structs::mat::{Mat1, Mat2, Mat3};
use traits::operations::{Inv, Det, ApproxEq};
use traits::structure::{Row, Col, BaseNum};

// some specializations:
impl<N: BaseNum + ApproxEq<N>> Inv for Mat1<N> {
    #[inline]
    fn inv_cpy(&self) -> Option<Mat1<N>> {
        let mut res = *self;
        if res.inv() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inv(&mut self) -> bool {
        if ApproxEq::approx_eq(&self.m11, &::zero()) {
            false
        }
        else {
            let _1: N = ::one();

            self.m11 = _1 / Det::det(self);
            true
        }
    }
}

impl<N: BaseNum + ApproxEq<N>> Inv for Mat2<N> {
    #[inline]
    fn inv_cpy(&self) -> Option<Mat2<N>> {
        let mut res = *self;
        if res.inv() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inv(&mut self) -> bool {
        let det = Det::det(self);

        if ApproxEq::approx_eq(&det, &::zero()) {
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

impl<N: BaseNum + ApproxEq<N>> Inv for Mat3<N> {
    #[inline]
    fn inv_cpy(&self) -> Option<Mat3<N>> {
        let mut res = *self;

        if res.inv() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inv(&mut self) -> bool {
        let minor_m12_m23 = self.m22 * self.m33 - self.m32 * self.m23;
        let minor_m11_m23 = self.m21 * self.m33 - self.m31 * self.m23;
        let minor_m11_m22 = self.m21 * self.m32 - self.m31 * self.m22;

        let det = self.m11 * minor_m12_m23 - self.m12 * minor_m11_m23 + self.m13 * minor_m11_m22;

        if ApproxEq::approx_eq(&det, &::zero()) {
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

impl<N: BaseNum> Det<N> for Mat1<N> {
    #[inline]
    fn det(&self) -> N {
        self.m11
    }
}

impl<N: BaseNum> Det<N> for Mat2<N> {
    #[inline]
    fn det(&self) -> N {
        self.m11 * self.m22 - self.m21 * self.m12
    }
}

impl<N: BaseNum> Det<N> for Mat3<N> {
    #[inline]
    fn det(&self) -> N {
        let minor_m12_m23 = self.m22 * self.m33 - self.m32 * self.m23;
        let minor_m11_m23 = self.m21 * self.m33 - self.m31 * self.m23;
        let minor_m11_m22 = self.m21 * self.m32 - self.m31 * self.m22;

        self.m11 * minor_m12_m23 - self.m12 * minor_m11_m23 + self.m13 * minor_m11_m22
    }
}

impl<N: Copy> Row<Vec3<N>> for Mat3<N> {
    #[inline]
    fn nrows(&self) -> uint {
        3
    }

    #[inline]
    fn row(&self, i: uint) -> Vec3<N> {
        match i {
            0 => Vec3::new(self.m11, self.m12, self.m13),
            1 => Vec3::new(self.m21, self.m22, self.m23),
            2 => Vec3::new(self.m31, self.m32, self.m33),
            _ => panic!(format!("Index out of range: 3d matrices do not have {} rows.",  i))
        }
    }

    #[inline]
    fn set_row(&mut self, i: uint, r: Vec3<N>) {
        match i {
            0 => {
                self.m11 = r.x;
                self.m12 = r.y;
                self.m13 = r.z;
            },
            1 => {
                self.m21 = r.x;
                self.m22 = r.y;
                self.m23 = r.z;
            },
            2 => {
                self.m31 = r.x;
                self.m32 = r.y;
                self.m33 = r.z;
            },
            _ => panic!(format!("Index out of range: 3d matrices do not have {} rows.",  i))

        }
    }
}

impl<N: Copy> Col<Vec3<N>> for Mat3<N> {
    #[inline]
    fn ncols(&self) -> uint {
        3
    }

    #[inline]
    fn col(&self, i: uint) -> Vec3<N> {
        match i {
            0 => Vec3::new(self.m11, self.m21, self.m31),
            1 => Vec3::new(self.m12, self.m22, self.m32),
            2 => Vec3::new(self.m13, self.m23, self.m33),
            _ => panic!(format!("Index out of range: 3d matrices do not have {} cols.", i))
        }
    }

    #[inline]
    fn set_col(&mut self, i: uint, r: Vec3<N>) {
        match i {
            0 => {
                self.m11 = r.x;
                self.m21 = r.y;
                self.m31 = r.z;
            },
            1 => {
                self.m12 = r.x;
                self.m22 = r.y;
                self.m32 = r.z;
            },
            2 => {
                self.m13 = r.x;
                self.m23 = r.y;
                self.m33 = r.z;
            },
            _ => panic!(format!("Index out of range: 3d matrices do not have {} cols.", i))

        }
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Mat3<N>, Mat3<N>> for Mat3<N> {
    #[inline]
    fn mul(self, right: Mat3<N>) -> Mat3<N> {
        Mat3::new(
            self.m11 * right.m11 + self.m12 * right.m21 + self.m13 * right.m31,
            self.m11 * right.m12 + self.m12 * right.m22 + self.m13 * right.m32,
            self.m11 * right.m13 + self.m12 * right.m23 + self.m13 * right.m33,

            self.m21 * right.m11 + self.m22 * right.m21 + self.m23 * right.m31,
            self.m21 * right.m12 + self.m22 * right.m22 + self.m23 * right.m32,
            self.m21 * right.m13 + self.m22 * right.m23 + self.m23 * right.m33,

            self.m31 * right.m11 + self.m32 * right.m21 + self.m33 * right.m31,
            self.m31 * right.m12 + self.m32 * right.m22 + self.m33 * right.m32,
            self.m31 * right.m13 + self.m32 * right.m23 + self.m33 * right.m33
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Mat2<N>, Mat2<N>> for Mat2<N> {
    #[inline(always)]
    fn mul(self, right: Mat2<N>) -> Mat2<N> {
        Mat2::new(
            self.m11 * right.m11 + self.m12 * right.m21,
            self.m11 * right.m12 + self.m12 * right.m22,

            self.m21 * right.m11 + self.m22 * right.m21,
            self.m21 * right.m12 + self.m22 * right.m22
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Vec3<N>, Vec3<N>> for Mat3<N> {
    #[inline(always)]
    fn mul(self, right: Vec3<N>) -> Vec3<N> {
        Vec3::new(
            self.m11 * right.x + self.m12 * right.y + self.m13 * right.z,
            self.m21 * right.x + self.m22 * right.y + self.m23 * right.z,
            self.m31 * right.x + self.m32 * right.y + self.m33 * right.z
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Mat3<N>, Vec3<N>> for Vec3<N> {
    #[inline(always)]
    fn mul(self, right: Mat3<N>) -> Vec3<N> {
        Vec3::new(
            self.x * right.m11 + self.y * right.m21 + self.z * right.m31,
            self.x * right.m12 + self.y * right.m22 + self.z * right.m32,
            self.x * right.m13 + self.y * right.m23 + self.z * right.m33
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Mat2<N>, Vec2<N>> for Vec2<N> {
    #[inline(always)]
    fn mul(self, right: Mat2<N>) -> Vec2<N> {
        Vec2::new(
            self.x * right.m11 + self.y * right.m21,
            self.x * right.m12 + self.y * right.m22
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Vec2<N>, Vec2<N>> for Mat2<N> {
    #[inline(always)]
    fn mul(self, right: Vec2<N>) -> Vec2<N> {
        Vec2::new(
            self.m11 * right.x + self.m12 * right.y,
            self.m21 * right.x + self.m22 * right.y
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Pnt3<N>, Pnt3<N>> for Mat3<N> {
    #[inline(always)]
    fn mul(self, right: Pnt3<N>) -> Pnt3<N> {
        Pnt3::new(
            self.m11 * right.x + self.m12 * right.y + self.m13 * right.z,
            self.m21 * right.x + self.m22 * right.y + self.m23 * right.z,
            self.m31 * right.x + self.m32 * right.y + self.m33 * right.z
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Mat3<N>, Pnt3<N>> for Pnt3<N> {
    #[inline(always)]
    fn mul(self, right: Mat3<N>) -> Pnt3<N> {
        Pnt3::new(
            self.x * right.m11 + self.y * right.m21 + self.z * right.m31,
            self.x * right.m12 + self.y * right.m22 + self.z * right.m32,
            self.x * right.m13 + self.y * right.m23 + self.z * right.m33
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Mat2<N>, Pnt2<N>> for Pnt2<N> {
    #[inline(always)]
    fn mul(self, right: Mat2<N>) -> Pnt2<N> {
        Pnt2::new(
            self.x * right.m11 + self.y * right.m21,
            self.x * right.m12 + self.y * right.m22
        )
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N>> Mul<Pnt2<N>, Pnt2<N>> for Mat2<N> {
    #[inline(always)]
    fn mul(self, right: Pnt2<N>) -> Pnt2<N> {
        Pnt2::new(
            self.m11 * right.x + self.m12 * right.y,
            self.m21 * right.x + self.m22 * right.y
        )
    }
}
