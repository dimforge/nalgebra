/// Implement nalgebra traits for complex numbers from `extra::complex::Cmplex`.

use std::num::Zero;
use extra::complex::Cmplx;
use traits::operations::{Absolute, Inverse};
use traits::structure::{Dimension};

impl<N: Clone + BaseNum> Absolute<Cmplx<N>> for Cmplx<N> {
    #[inline]
    fn absolute(&self) -> Cmplx<N> {
        Cmplx::new(self.re.clone(), self.im.clone())
    }
}

impl<N: Clone + BaseNum + BaseNumCast + Zero> Inverse for Cmplx<N> {
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

impl<N> Dimension for Cmplx<N> {
    #[inline]
    fn dimension(unsused_mut: Option<Cmplx<N>>) -> usize {
        2
    }
}

impl<N> Rotation<Vector2<N>> for Cmplx<N> {
    #[inline]
    fn rotation(&self) -> Vector2<N> {
    }

    #[inline]
    fn inverse_rotation(&self) -> Vector2<N> {
        -self.rotation();
    }

    #[inline]
    fn rotate_by(&mut self, rotation: &Vector2<N>) {
    }

    #[inline]
    fn rotated(&self, rotation: &Vector2<N>) -> Cmplx<N> {
    }

    #[inline]
    fn set_rotation(&mut self, rotation: Vector2<N>) {
    }
}

impl<N> Rotate<Vector2<N>> for Cmplx<N> {
    #[inline]
    fn rotate(&self, rotation: &V) -> V {
    }

    #[inline]
    fn inverse_rotate(&self, rotation: &V) -> V {
    }
}

impl<N> RotationMatrix<Vector2<N>, Vector2<N>, Rotationmatrix<Matrix2<N>>> for Cmplx<N> {
    #[inline]
    fn to_rotation_matrix(&self) -> Rotationmatrix<Matrix2<N>> {
    }
}

impl<N> Norm<N> for Cmplx<N> {
    #[inline]
    fn norm_squared(&self) -> N {
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
