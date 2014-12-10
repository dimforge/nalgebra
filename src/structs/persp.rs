use traits::structure::BaseFloat;
use structs::{Pnt3, Vec3, Mat4};

/// A 3D perspective projection stored without any matrix.
///
/// Reading or modifying its individual properties is cheap but applying the transformation is costly.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Show, Copy)]
pub struct Persp3<N> {
    aspect: N,
    fov:    N,
    znear:  N,
    zfar:   N
}

/// A 3D perspective projection stored as a 4D matrix.
///
/// Reading or modifying its individual properties is costly but applying the transformation is cheap.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Show, Copy)]
pub struct PerspMat3<N> {
    mat: Mat4<N>
}

impl<N: BaseFloat> Persp3<N> {
    /// Creates a new 3D perspective projection.
    pub fn new(aspect: N, fov: N, znear: N, zfar: N) -> Persp3<N> {
        assert!(!::is_zero(&(zfar - znear)));
        assert!(!::is_zero(&aspect));

        Persp3 {
            aspect: aspect,
            fov:    fov,
            znear:  znear,
            zfar:   zfar
        }
    }

    /// Builds a 4D projection matrix (using homogeneous coordinates) for this projection.
    pub fn to_mat(&self) -> Mat4<N> {
        self.to_persp_mat().mat
    }

    /// Build a `PerspMat3` representing this projection.
    pub fn to_persp_mat(&self) -> PerspMat3<N> {
        PerspMat3::new(self.aspect, self.fov, self.znear, self.zfar)
    }
}

impl<N: BaseFloat + Clone> Persp3<N> {
    /// Gets the `width / height` aspect ratio.
    #[inline]
    pub fn aspect(&self) -> N {
        self.aspect.clone()
    }

    /// Gets the field of view of the view frustrum.
    #[inline]
    pub fn fov(&self) -> N {
        self.fov.clone()
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

    /// Sets the field of view of the view frustrum.
    ///
    /// This method does not build any matrix.
    #[inline]
    pub fn set_fov(&mut self, fov: N) {
        self.fov = fov;
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

impl<N: BaseFloat> PerspMat3<N> {
    /// Creates a new persepctive matrix from the aspect ratio, field of view, and near/far planes.
    pub fn new(aspect: N, fov: N, znear: N, zfar: N) -> PerspMat3<N> {
        assert!(!::is_zero(&(znear - zfar)));
        assert!(!::is_zero(&aspect));

        let mat: Mat4<N> = ::one();

        let mut res = PerspMat3 { mat: mat };
        res.set_fov(fov);
        res.set_aspect(aspect);
        res.set_znear_and_zfar(znear, zfar);
        res.mat.m44 = ::zero();
        res.mat.m43 = ::one();

        res
    }

    /// Creates a new perspective projection matrix from a 4D matrix.
    ///
    /// This is unsafe because the input matrix is not checked to be a perspective projection.
    #[inline]
    pub unsafe fn new_with_mat(mat: Mat4<N>) -> PerspMat3<N> {
        PerspMat3 {
            mat: mat
        }
    }

    /// Returns a reference to the 4D matrix (using homogeneous coordinates) of this projection.
    #[inline]
    pub fn as_mat<'a>(&'a self) -> &'a Mat4<N> {
        &self.mat
    }

    /// Gets the `width / height` aspect ratio of the view frustrum.
    #[inline]
    pub fn aspect(&self) -> N {
        -self.mat.m22 / self.mat.m11
    }

    /// Gets the field of view of the view frustrum.
    #[inline]
    pub fn fov(&self) -> N {
        let _1: N = ::one();
        let _2 = _1 + _1;

        (_1 / self.mat.m22).atan() * _2
    }

    /// Gets the near plane offset of the view frustrum.
    #[inline]
    pub fn znear(&self) -> N {
        let _1: N = ::one();
        let _2 = _1 + _1;
        let ratio = (self.mat.m33 + _1) / (self.mat.m33 - _1);

        self.mat.m34 / (_2 * ratio) - self.mat.m34 / _2
    }

    /// Gets the far plane offset of the view frustrum.
    #[inline]
    pub fn zfar(&self) -> N {
        let _1: N = ::one();
        let _2 = _1 + _1;
        let ratio = (self.mat.m33 + _1) / (self.mat.m33 - _1);

        (self.mat.m34 - ratio * self.mat.m34) / _2
    }

    // FIXME: add a method to retriev znear and zfar at once ?

    /// Updates this projection matrix with a new `width / height` aspect ratio of the view
    /// frustrum.
    #[inline]
    pub fn set_aspect(&mut self, aspect: N) {
        assert!(!::is_zero(&aspect));
        self.mat.m11 = -self.mat.m22 / aspect;
    }

    /// Updates this projection with a new field of view of the view frustrum.
    #[inline]
    pub fn set_fov(&mut self, fov: N) {
        let _1: N = ::one();
        let _2 = _1 + _1;

        let old_m22  = self.mat.m22.clone();
        self.mat.m22 = _1 / (fov / _2).tan();
        self.mat.m11 = self.mat.m11 * (self.mat.m22 / old_m22);
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

        self.mat.m33 = -(zfar + znear) / (znear - zfar);
        self.mat.m34 = zfar * znear * _2 / (znear - zfar);
    }

    /// Projects a point.
    #[inline]
    pub fn project_pnt(&self, p: &Pnt3<N>) -> Pnt3<N> {
        let _1: N = ::one();
        let inv_denom = _1 / p.z;
        Pnt3::new(
            self.mat.m11 * p.x * inv_denom,
            self.mat.m22 * p.y * inv_denom,
            (self.mat.m33 * p.z + self.mat.m34) * inv_denom
        )
    }

    /// Projects a vector.
    #[inline]
    pub fn project_vec(&self, p: &Vec3<N>) -> Vec3<N> {
        let _1: N = ::one();
        let inv_denom = _1 / p.z;
        Vec3::new(
            self.mat.m11 * p.x * inv_denom,
            self.mat.m22 * p.y * inv_denom,
            self.mat.m33
        )
    }
}

impl<N: BaseFloat + Clone> PerspMat3<N> {
    /// Returns the 4D matrix (using homogeneous coordinates) of this projection.
    #[inline]
    pub fn to_mat<'a>(&'a self) -> Mat4<N> {
        self.mat.clone()
    }
}
