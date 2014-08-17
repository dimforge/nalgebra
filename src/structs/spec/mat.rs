use std::num::{Zero, One};
use structs::vec::{Vec2, Vec3, Vec2MulRhs, Vec3MulRhs};
use structs::mat::{Mat1, Mat2, Mat3, Mat3MulRhs, Mat2MulRhs};
use traits::operations::{Inv, Det, ApproxEq};
use traits::structure::{Row, Col};

// some specializations:
impl<N: Num + ApproxEq<N> + Clone> Inv for Mat1<N> {
    #[inline]
    fn inv_cpy(m: &Mat1<N>) -> Option<Mat1<N>> {
        let mut res = m.clone();

        if res.inv() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inv(&mut self) -> bool {
        if ApproxEq::approx_eq(&self.m11, &Zero::zero()) {
            false
        }
        else {
            let _1: N = One::one();

            self.m11 = _1 / Det::det(self);
            true
        }
    }
}

impl<N: Num + ApproxEq<N> + Clone> Inv for Mat2<N> {
    #[inline]
    fn inv_cpy(m: &Mat2<N>) -> Option<Mat2<N>> {
        let mut res = m.clone();

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

        if ApproxEq::approx_eq(&det, &Zero::zero()) {
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

impl<N: Num + ApproxEq<N> + Clone> Inv for Mat3<N> {
    #[inline]
    fn inv_cpy(m: &Mat3<N>) -> Option<Mat3<N>> {
        let mut res = m.clone();

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

        if ApproxEq::approx_eq(&det, &Zero::zero()) {
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

impl<N: Num + Clone> Det<N> for Mat1<N> {
    #[inline]
    fn det(m: &Mat1<N>) -> N {
        m.m11.clone()
    }
}

impl<N: Num> Det<N> for Mat2<N> {
    #[inline]
    fn det(m: &Mat2<N>) -> N {
        m.m11 * m.m22 - m.m21 * m.m12
    }
}

impl<N: Num> Det<N> for Mat3<N> {
    #[inline]
    fn det(m: &Mat3<N>) -> N {
        let minor_m12_m23 = m.m22 * m.m33 - m.m32 * m.m23;
        let minor_m11_m23 = m.m21 * m.m33 - m.m31 * m.m23;
        let minor_m11_m22 = m.m21 * m.m32 - m.m31 * m.m22;

        m.m11 * minor_m12_m23 - m.m12 * minor_m11_m23 + m.m13 * minor_m11_m22
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
            _ => fail!(format!("Index out of range: 3d matrices do not have {} rows.",  i))
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
            _ => fail!(format!("Index out of range: 3d matrices do not have {} rows.",  i))

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
            _ => fail!(format!("Index out of range: 3d matrices do not have {} cols.", i))
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
            _ => fail!(format!("Index out of range: 3d matrices do not have {} cols.", i))

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
