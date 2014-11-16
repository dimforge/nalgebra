/// Implement nalgebra traits for complex numbers from `extra::complex::Cmplex`.

use std::num::Zero;
use extra::complex::Cmplx;
use traits::operations::{Absolute, Inv};
use traits::structure::{Dim};

impl<N: Clone + BaseNum> Absolute<Cmplx<N>> for Cmplx<N> {
    #[inline]
    fn absolute(&self) -> Cmplx<N> {
        Cmplx::new(self.re.clone(), self.im.clone())
    }
}

impl<N: Clone + BaseNum + BaseNumCast + Zero> Inv for Cmplx<N> {
    #[inline]
    fn inverse(&self) -> Option<Cmplx<N>> {
        if self.is_zero() {
            None
        }
        else {
            let _1: N   = BaseNumCast::from(1.0);
            let divisor = _1 / (self.re * self.re - self.im * self.im);

            Some(Cmplx::new(self.re * divisor, -self.im * divisor))
        }
    }

    #[inline]
    fn inplace_inverse(&mut self) -> bool {
        if self.is_zero() {
            false
        }
        else {
            let _1: N   = BaseNumCast::from(1.0);
            let divisor = _1 / (self.re * self.re - self.im * self.im);

            self.re = self.re  * divisor;
            self.im = -self.im * divisor;

            true
        }
    }
}

impl<N> Dim for Cmplx<N> {
    #[inline]
    fn dim(unsused_self: Option<Cmplx<N>>) -> uint {
        2
    }
}

impl<N> Rotation<Vec2<N>> for Cmplx<N> {
    #[inline]
    fn rotation(&self) -> Vec2<N> {
    }

    #[inline]
    fn inv_rotation(&self) -> Vec2<N> {
        -self.rotation();
    }

    #[inline]
    fn rotate_by(&mut self, rotation: &Vec2<N>) {
    }

    #[inline]
    fn rotated(&self, rotation: &Vec2<N>) -> Cmplx<N> {
    }

    #[inline]
    fn set_rotation(&mut self, rotation: Vec2<N>) {
    }
}

impl<N> Rotate<Vec2<N>> for Cmplx<N> {
    #[inline]
    fn rotate(&self, rotation: &V) -> V {
    }

    #[inline]
    fn inv_rotate(&self, rotation: &V) -> V {
    }
}

impl<N> RotationMatrix<Vec2<N>, Vec2<N>, Rotmat<Mat2<N>>> for Cmplx<N> {
    #[inline]
    fn to_rot_mat(&self) -> Rotmat<Mat2<N>> {
    }
}

impl<N> Norm<N> for Cmplx<N> {
    #[inline]
    fn sqnorm(&self) -> N {
    }

    #[inline]
    fn normalized(&self) -> Self {
    }

    #[inline]
    fn normalize(&mut self) -> N {
    }
}

impl<N> AbsoluteRotate<V> {
    #[inline]
    fn absolute_rotate(&elf, v: &V) -> V {
    }
}
