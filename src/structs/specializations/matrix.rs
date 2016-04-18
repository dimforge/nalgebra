use std::ops::{Add, Mul, Neg, MulAssign};
use structs::vector::{Vector2, Vector3};
use structs::point::{Point2, Point3};
use structs::matrix::{Matrix1, Matrix2, Matrix3};
use traits::operations::{Inverse, Determinant, ApproxEq};
use traits::structure::{Row, Column, BaseNum};

// some specializations:
impl<N: BaseNum + ApproxEq<N>> Inverse for Matrix1<N> {
    #[inline]
    fn inverse(&self) -> Option<Matrix1<N>> {
        let mut res = *self;
        if res.inverse_mut() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inverse_mut(&mut self) -> bool {
        if ApproxEq::approx_eq(&self.m11, &::zero()) {
            false
        }
        else {
            let _1: N = ::one();

            self.m11 = _1 / Determinant::determinant(self);
            true
        }
    }
}

impl<N: BaseNum + Neg<Output = N> + ApproxEq<N>> Inverse for Matrix2<N> {
    #[inline]
    fn inverse(&self) -> Option<Matrix2<N>> {
        let mut res = *self;
        if res.inverse_mut() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inverse_mut(&mut self) -> bool {
        let determinant = Determinant::determinant(self);

        if ApproxEq::approx_eq(&determinant, &::zero()) {
            false
        }
        else {
            *self = Matrix2::new(
                self.m22 / determinant , -self.m12 / determinant,
                -self.m21 / determinant, self.m11 / determinant);

            true
        }
    }
}

impl<N: BaseNum + Neg<Output = N> + ApproxEq<N>> Inverse for Matrix3<N> {
    #[inline]
    fn inverse(&self) -> Option<Matrix3<N>> {
        let mut res = *self;

        if res.inverse_mut() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inverse_mut(&mut self) -> bool {
        let minor_m12_m23 = self.m22 * self.m33 - self.m32 * self.m23;
        let minor_m11_m23 = self.m21 * self.m33 - self.m31 * self.m23;
        let minor_m11_m22 = self.m21 * self.m32 - self.m31 * self.m22;

        let determinant = self.m11 * minor_m12_m23 - self.m12 * minor_m11_m23 + self.m13 * minor_m11_m22;

        if ApproxEq::approx_eq(&determinant, &::zero()) {
            false
        }
        else {
            *self = Matrix3::new(
                (minor_m12_m23 / determinant),
                ((self.m13 * self.m32 - self.m33 * self.m12) / determinant),
                ((self.m12 * self.m23 - self.m22 * self.m13) / determinant),

                (-minor_m11_m23 / determinant),
                ((self.m11 * self.m33 - self.m31 * self.m13) / determinant),
                ((self.m13 * self.m21 - self.m23 * self.m11) / determinant),

                (minor_m11_m22  / determinant),
                ((self.m12 * self.m31 - self.m32 * self.m11) / determinant),
                ((self.m11 * self.m22 - self.m21 * self.m12) / determinant)
                );

            true
        }
    }
}

impl<N: BaseNum> Determinant<N> for Matrix1<N> {
    #[inline]
    fn determinant(&self) -> N {
        self.m11
    }
}

impl<N: BaseNum> Determinant<N> for Matrix2<N> {
    #[inline]
    fn determinant(&self) -> N {
        self.m11 * self.m22 - self.m21 * self.m12
    }
}

impl<N: BaseNum> Determinant<N> for Matrix3<N> {
    #[inline]
    fn determinant(&self) -> N {
        let minor_m12_m23 = self.m22 * self.m33 - self.m32 * self.m23;
        let minor_m11_m23 = self.m21 * self.m33 - self.m31 * self.m23;
        let minor_m11_m22 = self.m21 * self.m32 - self.m31 * self.m22;

        self.m11 * minor_m12_m23 - self.m12 * minor_m11_m23 + self.m13 * minor_m11_m22
    }
}

impl<N: Copy> Row<Vector3<N>> for Matrix3<N> {
    #[inline]
    fn nrows(&self) -> usize {
        3
    }

    #[inline]
    fn row(&self, i: usize) -> Vector3<N> {
        match i {
            0 => Vector3::new(self.m11, self.m12, self.m13),
            1 => Vector3::new(self.m21, self.m22, self.m23),
            2 => Vector3::new(self.m31, self.m32, self.m33),
            _ => panic!(format!("Index out of range: 3d matrices do not have {} rows.",  i))
        }
    }

    #[inline]
    fn set_row(&mut self, i: usize, r: Vector3<N>) {
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

impl<N: Copy> Column<Vector3<N>> for Matrix3<N> {
    #[inline]
    fn ncols(&self) -> usize {
        3
    }

    #[inline]
    fn column(&self, i: usize) -> Vector3<N> {
        match i {
            0 => Vector3::new(self.m11, self.m21, self.m31),
            1 => Vector3::new(self.m12, self.m22, self.m32),
            2 => Vector3::new(self.m13, self.m23, self.m33),
            _ => panic!(format!("Index out of range: 3d matrices do not have {} cols.", i))
        }
    }

    #[inline]
    fn set_column(&mut self, i: usize, r: Vector3<N>) {
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

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Matrix3<N>> for Matrix3<N> {
    type Output = Matrix3<N>;

    #[inline]
    fn mul(self, right: Matrix3<N>) -> Matrix3<N> {
        Matrix3::new(
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

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Matrix2<N>> for Matrix2<N> {
    type Output = Matrix2<N>;

    #[inline(always)]
    fn mul(self, right: Matrix2<N>) -> Matrix2<N> {
        Matrix2::new(
            self.m11 * right.m11 + self.m12 * right.m21,
            self.m11 * right.m12 + self.m12 * right.m22,

            self.m21 * right.m11 + self.m22 * right.m21,
            self.m21 * right.m12 + self.m22 * right.m22
        )
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Vector3<N>> for Matrix3<N> {
    type Output = Vector3<N>;

    #[inline(always)]
    fn mul(self, right: Vector3<N>) -> Vector3<N> {
        Vector3::new(
            self.m11 * right.x + self.m12 * right.y + self.m13 * right.z,
            self.m21 * right.x + self.m22 * right.y + self.m23 * right.z,
            self.m31 * right.x + self.m32 * right.y + self.m33 * right.z
        )
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Matrix3<N>> for Vector3<N> {
    type Output = Vector3<N>;

    #[inline(always)]
    fn mul(self, right: Matrix3<N>) -> Vector3<N> {
        Vector3::new(
            self.x * right.m11 + self.y * right.m21 + self.z * right.m31,
            self.x * right.m12 + self.y * right.m22 + self.z * right.m32,
            self.x * right.m13 + self.y * right.m23 + self.z * right.m33
        )
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Matrix2<N>> for Vector2<N> {
    type Output = Vector2<N>;

    #[inline(always)]
    fn mul(self, right: Matrix2<N>) -> Vector2<N> {
        Vector2::new(
            self.x * right.m11 + self.y * right.m21,
            self.x * right.m12 + self.y * right.m22
        )
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Vector2<N>> for Matrix2<N> {
    type Output = Vector2<N>;

    #[inline(always)]
    fn mul(self, right: Vector2<N>) -> Vector2<N> {
        Vector2::new(
            self.m11 * right.x + self.m12 * right.y,
            self.m21 * right.x + self.m22 * right.y
        )
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Point3<N>> for Matrix3<N> {
    type Output = Point3<N>;

    #[inline(always)]
    fn mul(self, right: Point3<N>) -> Point3<N> {
        Point3::new(
            self.m11 * right.x + self.m12 * right.y + self.m13 * right.z,
            self.m21 * right.x + self.m22 * right.y + self.m23 * right.z,
            self.m31 * right.x + self.m32 * right.y + self.m33 * right.z
        )
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Matrix3<N>> for Point3<N> {
    type Output = Point3<N>;

    #[inline(always)]
    fn mul(self, right: Matrix3<N>) -> Point3<N> {
        Point3::new(
            self.x * right.m11 + self.y * right.m21 + self.z * right.m31,
            self.x * right.m12 + self.y * right.m22 + self.z * right.m32,
            self.x * right.m13 + self.y * right.m23 + self.z * right.m33
        )
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Matrix2<N>> for Point2<N> {
    type Output = Point2<N>;

    #[inline(always)]
    fn mul(self, right: Matrix2<N>) -> Point2<N> {
        Point2::new(
            self.x * right.m11 + self.y * right.m21,
            self.x * right.m12 + self.y * right.m22
        )
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> Mul<Point2<N>> for Matrix2<N> {
    type Output = Point2<N>;

    #[inline(always)]
    fn mul(self, right: Point2<N>) -> Point2<N> {
        Point2::new(
            self.m11 * right.x + self.m12 * right.y,
            self.m21 * right.x + self.m22 * right.y
        )
    }
}


macro_rules! impl_mul_assign_from_mul(
    ($tleft: ident, $tright: ident) => (
        impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N>> MulAssign<$tright<N>> for $tleft<N> {
            #[inline(always)]
            fn mul_assign(&mut self, right: $tright<N>) {
                // NOTE: there is probably no interesting optimization compared to the not-inplace
                // operation.
                *self = *self * right
            }
        }
    )
);

impl_mul_assign_from_mul!(Matrix3, Matrix3);
impl_mul_assign_from_mul!(Matrix2, Matrix2);

impl_mul_assign_from_mul!(Vector3, Matrix3);
impl_mul_assign_from_mul!(Vector2, Matrix2);
impl_mul_assign_from_mul!(Point3, Matrix3);
impl_mul_assign_from_mul!(Point2, Matrix2);
