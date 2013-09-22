use std::num::{Zero, One};
use vec::{Vec1, Vec2, Vec3, Norm, VecCast, UniformSphereSample, Cross, CrossMatrix, Basis};
use mat::{Mat3, Row};

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec1<N>> for Vec2<N> {
    #[inline]
    fn cross(&self, other : &Vec2<N>) -> Vec1<N> {
        Vec1::new(self.x * other.y - self.y * other.x)
    }
}

// FIXME: instead of returning a Vec2, define a Mat2x1 matrix?
impl<N: Neg<N> + Clone> CrossMatrix<Vec2<N>> for Vec2<N> {
    #[inline]
    fn cross_matrix(&self) -> Vec2<N> {
        Vec2::new(-self.y, self.x.clone())
    }
}

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec3<N>> for Vec3<N> {
    #[inline]
    fn cross(&self, other : &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    }
}

impl<N: Neg<N> + Zero + Clone> CrossMatrix<Mat3<N>> for Vec3<N> {
    #[inline]
    fn cross_matrix(&self) -> Mat3<N> {
        Mat3::new(
            Zero::zero()  , -self.z,        self.y.clone(),
            self.z.clone(), Zero::zero(),   -self.x,
            -self.y       , self.x.clone(), Zero::zero()
        )
    }
}

// FIXME: iplement this for all other vectors
impl<N: Clone> Row<Vec1<N>> for Vec2<N> {
    #[inline]
    fn num_rows(&self) -> uint {
        2
    }

    #[inline]
    fn row(&self, i: uint) -> Vec1<N> {
        match i {
            0 => Vec1::new(self.x.clone()),
            1 => Vec1::new(self.y.clone()),
            _ => fail!("Index out of range: 2d vectors do not have " + i.to_str() + " rows.")
        }
    }

    #[inline]
    fn set_row(&mut self, i: uint, r: Vec1<N>) {
        match i {
            0 => self.x = r.x,
            1 => self.y = r.x,
            _ => fail!("Index out of range: 2d vectors do not have " + i.to_str() + " rows.")

        }
    }
}

impl<N: One> Basis for Vec1<N> {
    #[inline(always)]
    fn canonical_basis(f: &fn(Vec1<N>) -> bool) {
        f(Vec1::new(One::one()));
    }

    #[inline(always)]
    fn orthonormal_subspace_basis(&self, _: &fn(Vec1<N>) -> bool ) { }
}

impl<N: Clone + One + Zero + Neg<N>> Basis for Vec2<N> {
    #[inline(always)]
    fn canonical_basis(f: &fn(Vec2<N>) -> bool) {
        if !f(Vec2::new(One::one(), Zero::zero())) { return };
        f(Vec2::new(Zero::zero(), One::one()));
    }

    #[inline]
    fn orthonormal_subspace_basis(&self, f: &fn(Vec2<N>) -> bool) {
        f(Vec2::new(-self.y, self.x.clone()));
    }
}

impl<N: Clone + Ord + Algebraic + Signed> Basis for Vec3<N> {
    #[inline(always)]
    fn canonical_basis(f: &fn(Vec3<N>) -> bool) {
        if !f(Vec3::new(One::one(), Zero::zero(), Zero::zero())) { return };
        if !f(Vec3::new(Zero::zero(), One::one(), Zero::zero())) { return };
        f(Vec3::new(Zero::zero(), Zero::zero(), One::one()));
    }

    #[inline(always)]
    fn orthonormal_subspace_basis(&self, f: &fn(Vec3<N>) -> bool) {
        let a = 
            if self.x.clone().abs() > self.y.clone().abs() {
                Vec3::new(self.z.clone(), Zero::zero(), -self.x).normalized()
            }
            else {
                Vec3::new(Zero::zero(), -self.z, self.y.clone()).normalized()
            };

        if !f(a.cross(self)) { return };
        f(a);
    }
}

// FIXME: this bad: this fixes definitly the number of samples…
static SAMPLES_2_F64: [Vec2<f64>, ..21] = [
    Vec2 { x: 1.0,         y: 0.0         },
    Vec2 { x: 0.95557281,  y: 0.29475517  },
    Vec2 { x: 0.82623877,  y: 0.56332006  },
    Vec2 { x: 0.6234898,   y: 0.78183148  },
    Vec2 { x: 0.36534102,  y: 0.93087375  },
    Vec2 { x: 0.07473009,  y: 0.9972038   },
    Vec2 { x: -0.22252093, y: 0.97492791  },
    Vec2 { x: -0.5,        y: 0.8660254   },
    Vec2 { x: -0.73305187, y: 0.68017274  },
    Vec2 { x: -0.90096887, y: 0.43388374  },
    Vec2 { x: -0.98883083, y: 0.14904227  },
    Vec2 { x: -0.98883083, y: -0.14904227 },
    Vec2 { x: -0.90096887, y: -0.43388374 },
    Vec2 { x: -0.73305187, y: -0.68017274 },
    Vec2 { x: -0.5,        y: -0.8660254  },
    Vec2 { x: -0.22252093, y: -0.97492791 },
    Vec2 { x: 0.07473009,  y: -0.9972038  },
    Vec2 { x: 0.36534102,  y: -0.93087375 },
    Vec2 { x: 0.6234898,   y: -0.78183148 },
    Vec2 { x: 0.82623877,  y: -0.56332006 },
    Vec2 { x: 0.95557281,  y: -0.29475517 },
];

// Those vectors come from bullet 3d
static SAMPLES_3_F64: [Vec3<f64>, ..42] = [
    Vec3 { x: 0.000000 , y: -0.000000, z: -1.000000 },
    Vec3 { x: 0.723608 , y: -0.525725, z: -0.447219 },
    Vec3 { x: -0.276388, y: -0.850649, z: -0.447219 },
    Vec3 { x: -0.894426, y: -0.000000, z: -0.447216 },
    Vec3 { x: -0.276388, y: 0.850649 , z: -0.447220 },
    Vec3 { x: 0.723608 , y: 0.525725 , z: -0.447219 },
    Vec3 { x: 0.276388 , y: -0.850649, z: 0.447220 },
    Vec3 { x: -0.723608, y: -0.525725, z: 0.447219 },
    Vec3 { x: -0.723608, y: 0.525725 , z: 0.447219 },
    Vec3 { x: 0.276388 , y: 0.850649 , z: 0.447219 },
    Vec3 { x: 0.894426 , y: 0.000000 , z: 0.447216 },
    Vec3 { x: -0.000000, y: 0.000000 , z: 1.000000 },
    Vec3 { x: 0.425323 , y: -0.309011, z: -0.850654 },
    Vec3 { x: -0.162456, y: -0.499995, z: -0.850654 },
    Vec3 { x: 0.262869 , y: -0.809012, z: -0.525738 },
    Vec3 { x: 0.425323 , y: 0.309011 , z: -0.850654 },
    Vec3 { x: 0.850648 , y: -0.000000, z: -0.525736 },
    Vec3 { x: -0.525730, y: -0.000000, z: -0.850652 },
    Vec3 { x: -0.688190, y: -0.499997, z: -0.525736 },
    Vec3 { x: -0.162456, y: 0.499995 , z: -0.850654 },
    Vec3 { x: -0.688190, y: 0.499997 , z: -0.525736 },
    Vec3 { x: 0.262869 , y: 0.809012 , z: -0.525738 },
    Vec3 { x: 0.951058 , y: 0.309013 , z: 0.000000 },
    Vec3 { x: 0.951058 , y: -0.309013, z: 0.000000 },
    Vec3 { x: 0.587786 , y: -0.809017, z: 0.000000 },
    Vec3 { x: 0.000000 , y: -1.000000, z: 0.000000 },
    Vec3 { x: -0.587786, y: -0.809017, z: 0.000000 },
    Vec3 { x: -0.951058, y: -0.309013, z: -0.000000 },
    Vec3 { x: -0.951058, y: 0.309013 , z: -0.000000 },
    Vec3 { x: -0.587786, y: 0.809017 , z: -0.000000 },
    Vec3 { x: -0.000000, y: 1.000000 , z: -0.000000 },
    Vec3 { x: 0.587786 , y: 0.809017 , z: -0.000000 },
    Vec3 { x: 0.688190 , y: -0.499997, z: 0.525736 },
    Vec3 { x: -0.262869, y: -0.809012, z: 0.525738 },
    Vec3 { x: -0.850648, y: 0.000000 , z: 0.525736 },
    Vec3 { x: -0.262869, y: 0.809012 , z: 0.525738 },
    Vec3 { x: 0.688190 , y: 0.499997 , z: 0.525736 },
    Vec3 { x: 0.525730 , y: 0.000000 , z: 0.850652 },
    Vec3 { x: 0.162456 , y: -0.499995, z: 0.850654 },
    Vec3 { x: -0.425323, y: -0.309011, z: 0.850654 },
    Vec3 { x: -0.425323, y: 0.309011 , z: 0.850654 },
    Vec3 { x: 0.162456 , y: 0.499995 , z: 0.850654 }
];

impl<N: One + Clone> UniformSphereSample for Vec1<N> {
    #[inline(always)]
    fn sample(f: &fn(Vec1<N>)) {
        f(One::one())
     }
}

impl<N: NumCast + Clone> UniformSphereSample for Vec2<N> {
    #[inline(always)]
    fn sample(f: &fn(Vec2<N>)) {
         for sample in SAMPLES_2_F64.iter() {
             f(VecCast::from(*sample))
         }
     }
}

impl<N: NumCast + Clone> UniformSphereSample for Vec3<N> {
    #[inline(always)]
    fn sample(f: &fn(Vec3<N>)) {
        for sample in SAMPLES_3_F64.iter() {
            f(VecCast::from(*sample))
        }
    }
}
