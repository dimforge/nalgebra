use base::{Scalar, Vector, DimName, Vector2, Vector3};
use storage::Storage;
use typenum::{self, Cmp, Greater};


macro_rules! impl_swizzle {
    ($(where $BaseDim: ident: $name: ident() -> $Result: ident[$($i: expr),*]);*) => {
        $(
            impl<N: Scalar, D: DimName, S: Storage<N, D>> Vector<N, D, S> {
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
    where U0: xx() -> Vector2[0, 0];
    where U1: xy() -> Vector2[0, 1];
    where U2: xz() -> Vector2[0, 2];
    where U1: yx() -> Vector2[1, 0];
    where U1: yy() -> Vector2[1, 1];
    where U1: yz() -> Vector2[1, 2];
    where U2: zx() -> Vector2[2, 0];
    where U2: zy() -> Vector2[2, 1];
    where U2: zz() -> Vector2[2, 2];

    where U0: xxx() -> Vector3[0, 0, 0];
    where U1: xxy() -> Vector3[0, 0, 1];
    where U2: xxz() -> Vector3[0, 0, 2];

    where U1: xyx() -> Vector3[0, 1, 0];
    where U1: xyy() -> Vector3[0, 1, 1];
    where U2: xyz() -> Vector3[0, 1, 2];

    where U2: xzx() -> Vector3[0, 2, 0];
    where U2: xzy() -> Vector3[0, 2, 1];
    where U2: xzz() -> Vector3[0, 2, 2];

    where U1: yxx() -> Vector3[1, 0, 0];
    where U1: yxy() -> Vector3[1, 0, 1];
    where U2: yxz() -> Vector3[1, 0, 2];

    where U1: yyx() -> Vector3[1, 1, 0];
    where U1: yyy() -> Vector3[1, 1, 1];
    where U2: yyz() -> Vector3[1, 1, 2];

    where U2: yzx() -> Vector3[1, 2, 0];
    where U2: yzy() -> Vector3[1, 2, 1];
    where U2: yzz() -> Vector3[1, 2, 2];

    where U2: zxx() -> Vector3[2, 0, 0];
    where U2: zxy() -> Vector3[2, 0, 1];
    where U2: zxz() -> Vector3[2, 0, 2];

    where U2: zyx() -> Vector3[2, 1, 0];
    where U2: zyy() -> Vector3[2, 1, 1];
    where U2: zyz() -> Vector3[2, 1, 2];

    where U2: zzx() -> Vector3[2, 2, 0];
    where U2: zzy() -> Vector3[2, 2, 1];
    where U2: zzz() -> Vector3[2, 2, 2]
);