use traits::structure::{BaseFloat, Cast};
use structs::{Point3, Vector3, Matrix4};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// A 3D orthographic projection stored without any matrix.
///
/// This flips the `z` axis and maps a axis-aligned cube to the unit cube with corners varying from
/// `(-1, -1, -1)` to `(1, 1, 1)`. Reading or modifying its individual properties is cheap but
/// applying the transformation is costly.
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Orthographic3<N> {
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
pub struct OrthographicMatrix3<N> {
    matrix: Matrix4<N>
}

impl<N: BaseFloat> Orthographic3<N> {
    /// Creates a new 3D orthographic projection.
    pub fn new(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> Orthographic3<N> {
        assert!(!::is_zero(&(zfar - znear)));
        assert!(!::is_zero(&(left - right)));
        assert!(!::is_zero(&(top - bottom)));

        Orthographic3 {
            left:   left,
            right:  right,
            bottom: bottom,
            top:    top,
            znear:  znear,
            zfar:   zfar
        }
    }

    /// Builds a 4D projection matrix (using homogeneous coordinates) for this projection.
    pub fn to_matrix(&self) -> Matrix4<N> {
        self.to_orthographic_matrix().matrix
    }

    /// Build a `OrthographicMatrix3` representing this projection.
    pub fn to_orthographic_matrix(&self) -> OrthographicMatrix3<N> {
        OrthographicMatrix3::new(self.left, self.right, self.bottom, self.top, self.znear, self.zfar)
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for Orthographic3<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> Orthographic3<N> {
        let left   = Arbitrary::arbitrary(g);
        let right  = reject(g, |x: &N| *x > left);
        let bottom = Arbitrary::arbitrary(g);
        let top    = reject(g, |x: &N| *x > bottom);
        let znear  = Arbitrary::arbitrary(g);
        let zfar   = reject(g, |x: &N| *x > znear);
        Orthographic3::new(left, right, bottom, top, znear, zfar)
    }
}

impl<N: BaseFloat> Orthographic3<N> {
    /// The smallest x-coordinate of the view cuboid.
    #[inline]
    pub fn left(&self) -> N {
        self.left
    }

    /// The largest x-coordinate of the view cuboid.
    #[inline]
    pub fn right(&self) -> N {
        self.right
    }

    /// The smallest y-coordinate of the view cuboid.
    #[inline]
    pub fn bottom(&self) -> N {
        self.bottom
    }

    /// The largest y-coordinate of the view cuboid.
    #[inline]
    pub fn top(&self) -> N {
        self.top
    }

    /// The near plane offset of the view cuboid.
    #[inline]
    pub fn znear(&self) -> N {
        self.znear
    }

    /// The far plane offset of the view cuboid.
    #[inline]
    pub fn zfar(&self) -> N {
        self.zfar
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
    pub fn project_point(&self, p: &Point3<N>) -> Point3<N> {
        // FIXME: optimize that
        self.to_orthographic_matrix().project_point(p)
    }

    /// Projects a vector.
    #[inline]
    pub fn project_vector(&self, p: &Vector3<N>) -> Vector3<N> {
        // FIXME: optimize that
        self.to_orthographic_matrix().project_vector(p)
    }
}

impl<N: BaseFloat> OrthographicMatrix3<N> {
    /// Creates a new orthographic projection matrix.
    pub fn new(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> OrthographicMatrix3<N> {
        assert!(left < right, "The left corner must be farther than the right corner.");
        assert!(bottom < top, "The top corner must be higher than the bottom corner.");
        assert!(znear < zfar, "The far plane must be farther than the near plane.");

        let matrix: Matrix4<N> = ::one();

        let mut res = OrthographicMatrix3 { matrix: matrix };
        res.set_left_and_right(left, right);
        res.set_bottom_and_top(bottom, top);
        res.set_znear_and_zfar(znear, zfar);

        res
    }

    /// Creates a new orthographic projection matrix from an aspect ratio and the vertical field of view.
    pub fn new_with_fov(aspect: N, vfov: N, znear: N, zfar: N) -> OrthographicMatrix3<N> {
        assert!(znear < zfar, "The far plane must be farther than the near plane.");
        assert!(!::is_zero(&aspect));

        let half: N = ::cast(0.5);
        let width  = zfar * (vfov * half).tan();
        let height = width / aspect;

        OrthographicMatrix3::new(-width * half, width * half, -height * half, height * half, znear, zfar)
    }

    /// Creates a new orthographic matrix from a 4D matrix.
    ///
    /// This is unsafe because the input matrix is not checked to be a orthographic projection.
    #[inline]
    pub unsafe fn new_with_matrix(matrix: Matrix4<N>) -> OrthographicMatrix3<N> {
        OrthographicMatrix3 {
            matrix: matrix
        }
    }

    /// Returns a reference to the 4D matrix (using homogeneous coordinates) of this projection.
    #[inline]
    pub fn as_matrix(&self) -> &Matrix4<N> {
        &self.matrix
    }

    /// The smallest x-coordinate of the view cuboid.
    #[inline]
    pub fn left(&self) -> N {
        (-::one::<N>() - self.matrix.m14) / self.matrix.m11
    }

    /// The largest x-coordinate of the view cuboid.
    #[inline]
    pub fn right(&self) -> N {
        (::one::<N>() - self.matrix.m14) / self.matrix.m11
    }

    /// The smallest y-coordinate of the view cuboid.
    #[inline]
    pub fn bottom(&self) -> N {
        (-::one::<N>() - self.matrix.m24) / self.matrix.m22
    }

    /// The largest y-coordinate of the view cuboid.
    #[inline]
    pub fn top(&self) -> N {
        (::one::<N>() - self.matrix.m24) / self.matrix.m22
    }

    /// The near plane offset of the view cuboid.
    #[inline]
    pub fn znear(&self) -> N {
        (::one::<N>() + self.matrix.m34) / self.matrix.m33
    }

    /// The far plane offset of the view cuboid.
    #[inline]
    pub fn zfar(&self) -> N {
        (-::one::<N>() + self.matrix.m34) / self.matrix.m33
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
        self.matrix.m11 = <N as Cast<f64>>::from(2.0) / (right - left);
        self.matrix.m14 = -(right + left) / (right - left);
    }

    /// Sets the view cuboid coordinates along the `y` axis.
    #[inline]
    pub fn set_bottom_and_top(&mut self, bottom: N, top: N) {
        assert!(bottom < top, "The top corner must be higher than the bottom corner.");
        self.matrix.m22 = <N as Cast<f64>>::from(2.0) / (top - bottom);
        self.matrix.m24 = -(top + bottom) / (top - bottom);
    }

    /// Sets the near and far plane offsets of the view cuboid.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        assert!(!::is_zero(&(zfar - znear)));
        self.matrix.m33 = -<N as Cast<f64>>::from(2.0) / (zfar - znear);
        self.matrix.m34 = -(zfar + znear) / (zfar - znear);
    }

    /// Projects a point.
    #[inline]
    pub fn project_point(&self, p: &Point3<N>) -> Point3<N> {
        Point3::new(
            self.matrix.m11 * p.x + self.matrix.m14,
            self.matrix.m22 * p.y + self.matrix.m24,
            self.matrix.m33 * p.z + self.matrix.m34
        )
    }

    /// Projects a vector.
    #[inline]
    pub fn project_vector(&self, p: &Vector3<N>) -> Vector3<N> {
        Vector3::new(
            self.matrix.m11 * p.x,
            self.matrix.m22 * p.y,
            self.matrix.m33 * p.z
        )
    }
}

impl<N: BaseFloat> OrthographicMatrix3<N> {
    /// Returns the 4D matrix (using homogeneous coordinates) of this projection.
    #[inline]
    pub fn to_matrix(&self) -> Matrix4<N> {
        self.matrix
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for OrthographicMatrix3<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> OrthographicMatrix3<N> {
        let x: Orthographic3<N> = Arbitrary::arbitrary(g);
        x.to_orthographic_matrix()
    }
}


/// Simple helper function for rejection sampling
#[cfg(feature="arbitrary")]
#[inline]
pub fn reject<G: Gen, F: FnMut(&T) -> bool, T: Arbitrary>(g: &mut G, f: F) -> T {
    use std::iter::repeat;
    repeat(()).map(|_| Arbitrary::arbitrary(g)).filter(f).next().unwrap()
}
