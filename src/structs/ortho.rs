use traits::structure::{BaseFloat, Cast};
use structs::{Pnt3, Vec3, Mat4};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// A 3D orthographic projection stored without any matrix.
///
/// Reading or modifying its individual properties is cheap but applying the transformation is costly.
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Ortho3<N> {
    width:  N,
    height: N,
    znear:  N,
    zfar:   N
}

/// A 3D orthographic projection stored as a 4D matrix.
///
/// Reading or modifying its individual properties is costly but applying the transformation is cheap.
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct OrthoMat3<N> {
    mat: Mat4<N>
}

impl<N: BaseFloat> Ortho3<N> {
    /// Creates a new 3D orthographic projection.
    pub fn new(width: N, height: N, znear: N, zfar: N) -> Ortho3<N> {
        assert!(!::is_zero(&(zfar - znear)));
        assert!(!::is_zero(&width));
        assert!(!::is_zero(&height));

        Ortho3 {
            width:  width,
            height: height,
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
        OrthoMat3::new(self.width, self.height, self.znear, self.zfar)
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for Ortho3<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> Ortho3<N> {
        let width = reject(g, |x| !::is_zero(x));
        let height = reject(g, |x| !::is_zero(x));
        let znear = Arbitrary::arbitrary(g);
        let zfar = reject(g, |&x: &N| !::is_zero(&(x - znear)));
        Ortho3::new(width, height, znear, zfar)
    }
}

impl<N: BaseFloat + Clone> Ortho3<N> {
    /// The width of the view cuboid.
    #[inline]
    pub fn width(&self) -> N {
        self.width.clone()
    }

    /// The height of the view cuboid.
    #[inline]
    pub fn height(&self) -> N {
        self.height.clone()
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

    /// Sets the width of the view cuboid.
    #[inline]
    pub fn set_width(&mut self, width: N) {
        self.width = width
    }

    /// Sets the height of the view cuboid.
    #[inline]
    pub fn set_height(&mut self, height: N) {
        self.height = height
    }

    /// Sets the near plane offset of the view cuboid.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        self.znear = znear
    }

    /// Sets the far plane offset of the view cuboid.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
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
    /// Creates a new orthographic projection matrix from the width, heihgt, znear and zfar planes of the view cuboid.
    pub fn new(width: N, height: N, znear: N, zfar: N) -> OrthoMat3<N> {
        assert!(!::is_zero(&(zfar - znear)));
        assert!(!::is_zero(&width));
        assert!(!::is_zero(&height));

        let mat: Mat4<N> = ::one();

        let mut res = OrthoMat3 { mat: mat };
        res.set_width(width);
        res.set_height(height);
        res.set_znear_and_zfar(znear, zfar);

        res
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

    /// The width of the view cuboid.
    #[inline]
    pub fn width(&self) -> N {
        <N as Cast<f64>>::from(2.0) / self.mat.m11
    }

    /// The height of the view cuboid.
    #[inline]
    pub fn height(&self) -> N {
        <N as Cast<f64>>::from(2.0) / self.mat.m22
    }

    /// The near plane offset of the view cuboid.
    #[inline]
    pub fn znear(&self) -> N {
        (self.mat.m34 + ::one()) / self.mat.m33
    }

    /// The far plane offset of the view cuboid.
    #[inline]
    pub fn zfar(&self) -> N {
        (self.mat.m34 - ::one()) / self.mat.m33
    }

    /// Sets the width of the view cuboid.
    #[inline]
    pub fn set_width(&mut self, width: N) {
        assert!(!::is_zero(&width));
        self.mat.m11 = <N as Cast<f64>>::from(2.0) / width;
    }

    /// Sets the height of the view cuboid.
    #[inline]
    pub fn set_height(&mut self, height: N) {
        assert!(!::is_zero(&height));
        self.mat.m22 = <N as Cast<f64>>::from(2.0) / height;
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
            self.mat.m11 * p.x,
            self.mat.m22 * p.y,
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
