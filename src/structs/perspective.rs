use traits::structure::BaseFloat;
use structs::{Point3, Vector3, Matrix4};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// A 3D perspective projection stored without any matrix.
///
/// This maps a frustrum cube to the unit cube with corners varying from `(-1, -1, -1)` to
/// `(1, 1, 1)`. Reading or modifying its individual properties is cheap but applying the
/// transformation is costly.
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Perspective3<N> {
    aspect: N,
    fovy:   N,
    znear:  N,
    zfar:   N
}

/// A 3D perspective projection stored as a 4D matrix.
///
/// This maps a frustrum to the unit cube with corners varying from `(-1, -1, -1)` to
/// `(1, 1, 1)`. Reading or modifying its individual properties is costly but applying the
/// transformation is cheap.
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct PerspectiveMatrix3<N> {
    matrix: Matrix4<N>
}

impl<N: BaseFloat> Perspective3<N> {
    /// Creates a new 3D perspective projection.
    pub fn new(aspect: N, fovy: N, znear: N, zfar: N) -> Perspective3<N> {
        assert!(!::is_zero(&(zfar - znear)));
        assert!(!::is_zero(&aspect));

        Perspective3 {
            aspect: aspect,
            fovy:   fovy,
            znear:  znear,
            zfar:   zfar
        }
    }

    /// Builds a 4D projection matrix (using homogeneous coordinates) for this projection.
    pub fn to_matrix(&self) -> Matrix4<N> {
        self.to_perspective_matrix().matrix
    }

    /// Build a `PerspectiveMatrix3` representing this projection.
    pub fn to_perspective_matrix(&self) -> PerspectiveMatrix3<N> {
        PerspectiveMatrix3::new(self.aspect, self.fovy, self.znear, self.zfar)
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for Perspective3<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> Perspective3<N> {
        use structs::orthographic::reject;
        let znear = Arbitrary::arbitrary(g);
        let zfar = reject(g, |&x: &N| !::is_zero(&(x - znear)));
        Perspective3::new(Arbitrary::arbitrary(g), Arbitrary::arbitrary(g), znear, zfar)
    }
}

impl<N: BaseFloat + Clone> Perspective3<N> {
    /// Gets the `width / height` aspect ratio.
    #[inline]
    pub fn aspect(&self) -> N {
        self.aspect.clone()
    }

    /// Gets the y field of view of the view frustrum.
    #[inline]
    pub fn fovy(&self) -> N {
        self.fovy.clone()
    }

    /// Gets the near plane offset of the view frustrum.
    #[inline]
    pub fn znear(&self) -> N {
        self.znear.clone()
    }

    /// Gets the far plane offset of the view frustrum.
    #[inline]
    pub fn zfar(&self) -> N {
        self.zfar.clone()
    }

    /// Sets the `width / height` aspect ratio of the view frustrum.
    ///
    /// This method does not build any matrix.
    #[inline]
    pub fn set_aspect(&mut self, aspect: N) {
        self.aspect = aspect;
    }

    /// Sets the y field of view of the view frustrum.
    ///
    /// This method does not build any matrix.
    #[inline]
    pub fn set_fovy(&mut self, fovy: N) {
        self.fovy = fovy;
    }

    /// Sets the near plane offset of the view frustrum.
    ///
    /// This method does not build any matrix.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        self.znear = znear;
    }

    /// Sets the far plane offset of the view frustrum.
    ///
    /// This method does not build any matrix.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        self.zfar = zfar;
    }

    /// Projects a point.
    #[inline]
    pub fn project_point(&self, p: &Point3<N>) -> Point3<N> {
        // FIXME: optimize that
        self.to_perspective_matrix().project_point(p)
    }

    /// Projects a vector.
    #[inline]
    pub fn project_vector(&self, p: &Vector3<N>) -> Vector3<N> {
        // FIXME: optimize that
        self.to_perspective_matrix().project_vector(p)
    }
}

impl<N: BaseFloat> PerspectiveMatrix3<N> {
    /// Creates a new perspective matrix from the aspect ratio, y field of view, and near/far planes.
    pub fn new(aspect: N, fovy: N, znear: N, zfar: N) -> PerspectiveMatrix3<N> {
        assert!(!::is_zero(&(znear - zfar)));
        assert!(!::is_zero(&aspect));

        let matrix: Matrix4<N> = ::one();

        let mut res = PerspectiveMatrix3 { matrix: matrix };
        res.set_fovy(fovy);
        res.set_aspect(aspect);
        res.set_znear_and_zfar(znear, zfar);
        res.matrix.m44 = ::zero();
        res.matrix.m43 = -::one::<N>();

        res
    }

    /// Creates a new perspective projection matrix from a 4D matrix.
    ///
    /// This is unsafe because the input matrix is not checked to be a perspective projection.
    #[inline]
    pub unsafe fn new_with_matrix(matrix: Matrix4<N>) -> PerspectiveMatrix3<N> {
        PerspectiveMatrix3 {
            matrix: matrix
        }
    }

    /// Returns a reference to the 4D matrix (using homogeneous coordinates) of this projection.
    #[inline]
    pub fn as_matrix<'a>(&'a self) -> &'a Matrix4<N> {
        &self.matrix
    }

    /// Gets the `width / height` aspect ratio of the view frustrum.
    #[inline]
    pub fn aspect(&self) -> N {
        self.matrix.m22 / self.matrix.m11
    }

    /// Gets the y field of view of the view frustrum.
    #[inline]
    pub fn fovy(&self) -> N {
        let _1: N = ::one();
        let _2 = _1 + _1;

        (_1 / self.matrix.m22).atan() * _2
    }

    /// Gets the near plane offset of the view frustrum.
    #[inline]
    pub fn znear(&self) -> N {
        let _1: N = ::one();
        let _2 = _1 + _1;
        let ratio = (-self.matrix.m33 + _1) / (-self.matrix.m33 - _1);

        self.matrix.m34 / (_2 * ratio) - self.matrix.m34 / _2
    }

    /// Gets the far plane offset of the view frustrum.
    #[inline]
    pub fn zfar(&self) -> N {
        let _1: N = ::one();
        let _2 = _1 + _1;
        let ratio = (-self.matrix.m33 + _1) / (-self.matrix.m33 - _1);

        (self.matrix.m34 - ratio * self.matrix.m34) / _2
    }

    // FIXME: add a method to retrieve znear and zfar simultaneously?

    /// Updates this projection matrix with a new `width / height` aspect ratio of the view
    /// frustrum.
    #[inline]
    pub fn set_aspect(&mut self, aspect: N) {
        assert!(!::is_zero(&aspect));
        self.matrix.m11 = self.matrix.m22 / aspect;
    }

    /// Updates this projection with a new y field of view of the view frustrum.
    #[inline]
    pub fn set_fovy(&mut self, fovy: N) {
        let _1: N = ::one();
        let _2 = _1 + _1;

        let old_m22  = self.matrix.m22.clone();
        self.matrix.m22 = _1 / (fovy / _2).tan();
        self.matrix.m11 = self.matrix.m11 * (self.matrix.m22 / old_m22);
    }

    /// Updates this projection matrix with a new near plane offset of the view frustrum.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this projection matrix with a new far plane offset of the view frustrum.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this projection matrix with new near and far plane offsets of the view frustrum.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        let _1: N = ::one();
        let _2 = _1 + _1;

        self.matrix.m33 = (zfar + znear) / (znear - zfar);
        self.matrix.m34 = zfar * znear * _2 / (znear - zfar);
    }

    /// Projects a point.
    #[inline]
    pub fn project_point(&self, p: &Point3<N>) -> Point3<N> {
        let _1: N = ::one();
        let inverse_denom = -_1 / p.z;
        Point3::new(
            self.matrix.m11 * p.x * inverse_denom,
            self.matrix.m22 * p.y * inverse_denom,
            (self.matrix.m33 * p.z + self.matrix.m34) * inverse_denom
        )
    }

    /// Projects a vector.
    #[inline]
    pub fn project_vector(&self, p: &Vector3<N>) -> Vector3<N> {
        let _1: N = ::one();
        let inverse_denom = -_1 / p.z;
        Vector3::new(
            self.matrix.m11 * p.x * inverse_denom,
            self.matrix.m22 * p.y * inverse_denom,
            self.matrix.m33
        )
    }
}

impl<N: BaseFloat + Clone> PerspectiveMatrix3<N> {
    /// Returns the 4D matrix (using homogeneous coordinates) of this projection.
    #[inline]
    pub fn to_matrix<'a>(&'a self) -> Matrix4<N> {
        self.matrix.clone()
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for PerspectiveMatrix3<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> PerspectiveMatrix3<N> {
        let x: Perspective3<N> = Arbitrary::arbitrary(g);
        x.to_perspective_matrix()
    }
}
