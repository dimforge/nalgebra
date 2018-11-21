use base::allocator::Allocator;
use base::{DefaultAllocator, DimName, Scalar};
use geometry::{Point, Point2, Point3};
use typenum::{self, Cmp, Greater};

macro_rules! impl_swizzle {
    ($(where $BaseDim: ident: $name: ident() -> $Result: ident[$($i: expr),*]);*) => {
        $(
            impl<N: Scalar, D: DimName> Point<N, D>
            where DefaultAllocator: Allocator<N, D>
            {
                /// Builds a new vector from components of `self`.
                #[inline]
                pub fn $name(&self) -> $Result<N>
                    where D::Value: Cmp<typenum::$BaseDim, Output=Greater> {
                    $Result::new($(self[$i]),*)
                }
            }
        )*
    }
}

impl_swizzle!(
    where U0: xx() -> Point2[0, 0];
    where U1: xy() -> Point2[0, 1];
    where U2: xz() -> Point2[0, 2];
    where U1: yx() -> Point2[1, 0];
    where U1: yy() -> Point2[1, 1];
    where U1: yz() -> Point2[1, 2];
    where U2: zx() -> Point2[2, 0];
    where U2: zy() -> Point2[2, 1];
    where U2: zz() -> Point2[2, 2];

    where U0: xxx() -> Point3[0, 0, 0];
    where U1: xxy() -> Point3[0, 0, 1];
    where U2: xxz() -> Point3[0, 0, 2];

    where U1: xyx() -> Point3[0, 1, 0];
    where U1: xyy() -> Point3[0, 1, 1];
    where U2: xyz() -> Point3[0, 1, 2];

    where U2: xzx() -> Point3[0, 2, 0];
    where U2: xzy() -> Point3[0, 2, 1];
    where U2: xzz() -> Point3[0, 2, 2];

    where U1: yxx() -> Point3[1, 0, 0];
    where U1: yxy() -> Point3[1, 0, 1];
    where U2: yxz() -> Point3[1, 0, 2];

    where U1: yyx() -> Point3[1, 1, 0];
    where U1: yyy() -> Point3[1, 1, 1];
    where U2: yyz() -> Point3[1, 1, 2];

    where U2: yzx() -> Point3[1, 2, 0];
    where U2: yzy() -> Point3[1, 2, 1];
    where U2: yzz() -> Point3[1, 2, 2];

    where U2: zxx() -> Point3[2, 0, 0];
    where U2: zxy() -> Point3[2, 0, 1];
    where U2: zxz() -> Point3[2, 0, 2];

    where U2: zyx() -> Point3[2, 1, 0];
    where U2: zyy() -> Point3[2, 1, 1];
    where U2: zyz() -> Point3[2, 1, 2];

    where U2: zzx() -> Point3[2, 2, 0];
    where U2: zzy() -> Point3[2, 2, 1];
    where U2: zzz() -> Point3[2, 2, 2]
);
