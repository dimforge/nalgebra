use traits::structure::{BaseFloat, Cast};
use structs::{Pnt3, Vec3, Mat4};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// A 3D orthographic projection stored without any matrix.
///
/// This flips the `z` axis and maps a axis-aligned cube to the unit cube with corners varying from
/// `(-1, -1, -1)` to `(1, 1, 1)`. Reading or modifying its individual properties is cheap but
/// applying the transformation is costly.
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Ortho3<N> {
    left:   N,
    right:  N,
    bottom: N,
    top:    N,
    znear:  N,
    zfar:   N
}

/// A 3D orthographic projection stored as a 4D matrix.
///
/// This flips the `z` axis and maps a axis-aligned cube to the unit cube with corners varying from
/// `(-1, -1, -1)` to `(1, 1, 1)`. Reading or modifying its individual properties is costly but
/// applying the transformation is cheap.
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct OrthoMat3<N> {
    mat: Mat4<N>
}

impl<N: BaseFloat> Ortho3<N> {
    /// Creates a new 3D orthographic projection.
    pub fn new(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> Ortho3<N> {
        assert!(!::is_zero(&(zfar - znear)));
        assert!(!::is_zero(&(left - right)));
        assert!(!::is_zero(&(top - bottom)));

        Ortho3 {
            left:   left,
            right:  right,
            bottom: bottom,
            top:    top,
            znear:  znear,
            zfar:   zfar
        }
    }

    /// Builds a 4D projection matrix (using homogeneous coordinates) for this projection.
    pub fn to_mat(&self) -> Mat4<N> {
        self.to_persp_mat().mat
    }

    /// Build a `OrthoMat3` representing this projection.
    pub fn to_persp_mat(&self) -> OrthoMat3<N> {
        OrthoMat3::new(self.left, self.right, self.bottom, self.top, self.znear, self.zfar)
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for Ortho3<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> Ortho3<N> {
        let left   = Arbitrary::arbitrary(g);
        let right  = reject(g, |x: &N| *x > left);
        let bottom = Arbitrary::arbitrary(g);
        let top    = reject(g, |x: &N| *x > bottom);
        let znear  = Arbitrary::arbitrary(g);
        let zfar   = reject(g, |x: &N| *x > znear);
        Ortho3::new(left, right, bottom, top, znear, zfar)
    }
}

impl<N: BaseFloat + Clone> Ortho3<N> {
    /// The smallest x-coordinate of the view cuboid.
    #[inline]
    pub fn left(&self) -> N {
        self.left.clone()
    }

    /// The largest x-coordinate of the view cuboid.
    #[inline]
    pub fn right(&self) -> N {
        self.right.clone()
    }

    /// The smallest y-coordinate of the view cuboid.
    #[inline]
    pub fn bottom(&self) -> N {
        self.bottom.clone()
    }

    /// The largest y-coordinate of the view cuboid.
    #[inline]
    pub fn top(&self) -> N {
        self.top.clone()
    }

    /// The near plane offset of the view cuboid.
    #[inline]
    pub fn znear(&self) -> N {
        self.znear.clone()
    }

    /// The far plane offset of the view cuboid.
    #[inline]
    pub fn zfar(&self) -> N {
        self.zfar.clone()
    }

    /// Sets the smallest x-coordinate of the view cuboid.
    #[inline]
    pub fn set_left(&mut self, left: N) {
        assert!(left < self.right, "The left corner must be farther than the right corner.");
        self.left = left 
    }

    /// Sets the largest x-coordinate of the view cuboid.
    #[inline]
    pub fn set_right(&mut self, right: N) {
        assert!(right > self.left, "The left corner must be farther than the right corner.");
        self.right = right
    }

    /// Sets the smallest y-coordinate of the view cuboid.
    #[inline]
    pub fn set_bottom(&mut self, bottom: N) {
        assert!(bottom < self.top, "The top corner must be higher than the bottom corner.");
        self.bottom = bottom
    }

    /// Sets the largest y-coordinate of the view cuboid.
    #[inline]
    pub fn set_top(&mut self, top: N) {
        assert!(top > self.bottom, "The top corner must be higher than the left corner.");
        self.top = top
    }

    /// Sets the near plane offset of the view cuboid.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        assert!(znear < self.zfar, "The far plane must be farther than the near plane.");
        self.znear = znear
    }

    /// Sets the far plane offset of the view cuboid.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        assert!(zfar > self.znear, "The far plane must be farther than the near plane.");
        self.zfar = zfar
    }

    /// Projects a point.
    #[inline]
    pub fn project_pnt(&self, p: &Pnt3<N>) -> Pnt3<N> {
        // FIXME: optimize that
        self.to_persp_mat().project_pnt(p)
    }

    /// Projects a vector.
    #[inline]
    pub fn project_vec(&self, p: &Vec3<N>) -> Vec3<N> {
        // FIXME: optimize that
        self.to_persp_mat().project_vec(p)
    }
}

impl<N: BaseFloat> OrthoMat3<N> {
    /// Creates a new orthographic projection matrix.
    pub fn new(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> OrthoMat3<N> {
        assert!(left < right, "The left corner must be farther than the right corner.");
        assert!(bottom < top, "The top corner must be higher than the bottom corner.");
        assert!(znear < zfar, "The far plane must be farther than the near plane.");

        let mat: Mat4<N> = ::one();

        let mut res = OrthoMat3 { mat: mat };
        res.set_left_and_right(left, right);
        res.set_bottom_and_top(bottom, top);
        res.set_znear_and_zfar(znear, zfar);

        res
    }

    /// Creates a new orthographic projection matrix from an aspect ratio and the vertical field of view.
    pub fn new_with_fov(aspect: N, vfov: N, znear: N, zfar: N) -> OrthoMat3<N> {
        assert!(znear < zfar, "The far plane must be farther than the near plane.");
        assert!(!::is_zero(&aspect));

        let _1: N = ::one();
        let _2 = _1 + _1;
        let width  = zfar * (vfov / _2).tan();
        let height = width / aspect;

        OrthoMat3::new(-width / _2, width / _2, -height / _2, height / _2, znear, zfar)
    }

    /// Creates a new orthographic matrix from a 4D matrix.
    ///
    /// This is unsafe because the input matrix is not checked to be a orthographic projection.
    #[inline]
    pub unsafe fn new_with_mat(mat: Mat4<N>) -> OrthoMat3<N> {
        OrthoMat3 {
            mat: mat
        }
    }

    /// Returns a reference to the 4D matrix (using homogeneous coordinates) of this projection.
    #[inline]
    pub fn as_mat<'a>(&'a self) -> &'a Mat4<N> {
        &self.mat
    }

    /// The smallest x-coordinate of the view cuboid.
    #[inline]
    pub fn left(&self) -> N {
        (-::one::<N>() - self.mat.m14) / self.mat.m11
    }

    /// The largest x-coordinate of the view cuboid.
    #[inline]
    pub fn right(&self) -> N {
        (::one::<N>() - self.mat.m14) / self.mat.m11
    }

    /// The smallest y-coordinate of the view cuboid.
    #[inline]
    pub fn bottom(&self) -> N {
        (-::one::<N>() - self.mat.m24) / self.mat.m22
    }

    /// The largest y-coordinate of the view cuboid.
    #[inline]
    pub fn top(&self) -> N {
        (::one::<N>() - self.mat.m24) / self.mat.m22
    }

    /// The near plane offset of the view cuboid.
    #[inline]
    pub fn znear(&self) -> N {
        (::one::<N>() + self.mat.m34) / self.mat.m33
    }

    /// The far plane offset of the view cuboid.
    #[inline]
    pub fn zfar(&self) -> N {
        (-::one::<N>() + self.mat.m34) / self.mat.m33
    }

    /// Sets the smallest x-coordinate of the view cuboid.
    #[inline]
    pub fn set_left(&mut self, left: N) {
        let right = self.right();
        self.set_left_and_right(left, right);
    }

    /// Sets the largest x-coordinate of the view cuboid.
    #[inline]
    pub fn set_right(&mut self, right: N) {
        let left = self.left();
        self.set_left_and_right(left, right);
    }

    /// Sets the smallest y-coordinate of the view cuboid.
    #[inline]
    pub fn set_bottom(&mut self, bottom: N) {
        let top = self.top();
        self.set_bottom_and_top(bottom, top);
    }

    /// Sets the largest y-coordinate of the view cuboid.
    #[inline]
    pub fn set_top(&mut self, top: N) {
        let bottom = self.bottom();
        self.set_bottom_and_top(bottom, top);
    }

    /// Sets the near plane offset of the view cuboid.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Sets the far plane offset of the view cuboid.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Sets the view cuboid coordinates along the `x` axis.
    #[inline]
    pub fn set_left_and_right(&mut self, left: N, right: N) {
        assert!(left < right, "The left corner must be farther than the right corner.");
        self.mat.m11 = <N as Cast<f64>>::from(2.0) / (right - left);
        self.mat.m14 = -(right + left) / (right - left);
    }

    /// Sets the view cuboid coordinates along the `y` axis.
    #[inline]
    pub fn set_bottom_and_top(&mut self, bottom: N, top: N) {
        assert!(bottom < top, "The top corner must be higher than the bottom corner.");
        self.mat.m22 = <N as Cast<f64>>::from(2.0) / (top - bottom);
        self.mat.m24 = -(top + bottom) / (top - bottom);
    }

    /// Sets the near and far plane offsets of the view cuboid.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        assert!(!::is_zero(&(zfar - znear)));
        self.mat.m33 = -<N as Cast<f64>>::from(2.0) / (zfar - znear);
        self.mat.m34 = -(zfar + znear) / (zfar - znear);
    }

    /// Projects a point.
    #[inline]
    pub fn project_pnt(&self, p: &Pnt3<N>) -> Pnt3<N> {
        Pnt3::new(
            self.mat.m11 * p.x + self.mat.m14,
            self.mat.m22 * p.y + self.mat.m24,
            self.mat.m33 * p.z + self.mat.m34
        )
    }

    /// Projects a vector.
    #[inline]
    pub fn project_vec(&self, p: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            self.mat.m11 * p.x,
            self.mat.m22 * p.y,
            self.mat.m33 * p.z
        )
    }
}

impl<N: BaseFloat + Clone> OrthoMat3<N> {
    /// Returns the 4D matrix (using homogeneous coordinates) of this projection.
    #[inline]
    pub fn to_mat<'a>(&'a self) -> Mat4<N> {
        self.mat.clone()
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for OrthoMat3<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> OrthoMat3<N> {
        let x: Ortho3<N> = Arbitrary::arbitrary(g);
        x.to_persp_mat()
    }
}


/// Simple helper function for rejection sampling
#[cfg(feature="arbitrary")]
#[inline]
pub fn reject<G: Gen, F: FnMut(&T) -> bool, T: Arbitrary>(g: &mut G, f: F) -> T {
    use std::iter::repeat;
    repeat(()).map(|_| Arbitrary::arbitrary(g)).filter(f).next().unwrap()
}
