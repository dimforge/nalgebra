use traits::structure::{Cast, Row, Basis, BaseFloat, Zero, One};
use traits::geometry::{Norm, Cross, CrossMatrix, UniformSphereSample};
use structs::vec::{Vec1, Vec2, Vec3, Vec4};
use structs::mat::Mat3;

impl<N: Copy + Mul<N, N> + Sub<N, N>> Cross<Vec1<N>> for Vec2<N> {
    #[inline]
    fn cross(&self, other: &Vec2<N>) -> Vec1<N> {
        Vec1::new(self.x * other.y - self.y * other.x)
    }
}

// FIXME: instead of returning a Vec2, define a Mat2x1 matrix?
impl<N: Neg<N> + Copy> CrossMatrix<Vec2<N>> for Vec2<N> {
    #[inline]
    fn cross_matrix(&self) -> Vec2<N> {
        Vec2::new(-self.y, self.x)
    }
}

impl<N: Copy + Mul<N, N> + Sub<N, N>> Cross<Vec3<N>> for Vec3<N> {
    #[inline]
    fn cross(&self, other: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    }
}

impl<N: Neg<N> + Zero + Copy> CrossMatrix<Mat3<N>> for Vec3<N> {
    #[inline]
    fn cross_matrix(&self) -> Mat3<N> {
        Mat3::new(
            ::zero(), -self.z,  self.y,
            self.z,   ::zero(), -self.x,
            -self.y,  self.x,   ::zero()
        )
    }
}

// FIXME: implement this for all other vectors
impl<N: Copy> Row<Vec1<N>> for Vec2<N> {
    #[inline]
    fn nrows(&self) -> uint {
        2
    }

    #[inline]
    fn row(&self, i: uint) -> Vec1<N> {
        match i {
            0 => Vec1::new(self.x),
            1 => Vec1::new(self.y),
            _ => panic!(format!("Index out of range: 2d vectors do not have {} rows. ", i))
        }
    }

    #[inline]
    fn set_row(&mut self, i: uint, r: Vec1<N>) {
        match i {
            0 => self.x = r.x,
            1 => self.y = r.x,
            _ => panic!(format!("Index out of range: 2d vectors do not have {} rows.", i))

        }
    }
}

impl<N: One> Basis for Vec1<N> {
    #[inline(always)]
    fn canonical_basis(f: |Vec1<N>| -> bool) {
        f(Vec1::new(::one()));
    }

    #[inline(always)]
    fn orthonormal_subspace_basis(_: &Vec1<N>, _: |Vec1<N>| -> bool) { }

    #[inline]
    fn canonical_basis_element(i: uint) -> Option<Vec1<N>> {
        if i == 0 {
            Some(Vec1::new(::one()))
        }
        else {
            None
        }
    }
}

impl<N: Copy + One + Zero + Neg<N>> Basis for Vec2<N> {
    #[inline(always)]
    fn canonical_basis(f: |Vec2<N>| -> bool) {
        if !f(Vec2::new(::one(), ::zero())) { return };
        f(Vec2::new(::zero(), ::one()));
    }

    #[inline]
    fn orthonormal_subspace_basis(n: &Vec2<N>, f: |Vec2<N>| -> bool) {
        f(Vec2::new(-n.y, n.x));
    }

    #[inline]
    fn canonical_basis_element(i: uint) -> Option<Vec2<N>> {
        if i == 0 {
            Some(Vec2::new(::one(), ::zero()))
        }
        else if i == 1 {
            Some(Vec2::new(::zero(), ::one()))
        }
        else {
            None
        }
    }
}

impl<N: BaseFloat> Basis for Vec3<N> {
    #[inline(always)]
    fn canonical_basis(f: |Vec3<N>| -> bool) {
        if !f(Vec3::new(::one(), ::zero(), ::zero())) { return };
        if !f(Vec3::new(::zero(), ::one(), ::zero())) { return };
        f(Vec3::new(::zero(), ::zero(), ::one()));
    }

    #[inline(always)]
    fn orthonormal_subspace_basis(n: &Vec3<N>, f: |Vec3<N>| -> bool) {
        let a = 
            if n.x.abs() > n.y.abs() {
                Norm::normalize_cpy(&Vec3::new(n.z, ::zero(), -n.x))
            }
            else {
                Norm::normalize_cpy(&Vec3::new(::zero(), -n.z, n.y))
            };

        if !f(Cross::cross(&a, n)) { return };
        f(a);
    }

    #[inline]
    fn canonical_basis_element(i: uint) -> Option<Vec3<N>> {
        if i == 0 {
            Some(Vec3::new(::one(), ::zero(), ::zero()))
        }
        else if i == 1 {
            Some(Vec3::new(::zero(), ::one(), ::zero()))
        }
        else if i == 2 {
            Some(Vec3::new(::zero(), ::zero(), ::one()))
        }
        else {
            None
        }
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

impl<N: One + Copy> UniformSphereSample for Vec1<N> {
    #[inline(always)]
    fn sample(f: |Vec1<N>| -> ()) {
        f(::one())
     }
}

impl<N: Cast<f64> + Copy> UniformSphereSample for Vec2<N> {
    #[inline(always)]
    fn sample(f: |Vec2<N>| -> ()) {
         for sample in SAMPLES_2_F64.iter() {
             f(Cast::from(*sample))
         }
     }
}

impl<N: Cast<f64> + Copy> UniformSphereSample for Vec3<N> {
    #[inline(always)]
    fn sample(f: |Vec3<N>| -> ()) {
        for sample in SAMPLES_3_F64.iter() {
            f(Cast::from(*sample))
        }
    }
}

impl<N: Cast<f64> + Copy> UniformSphereSample for Vec4<N> {
    #[inline(always)]
    fn sample(_: |Vec4<N>| -> ()) {
        panic!("UniformSphereSample::<Vec4<N>>::sample : Not yet implemented.")
        // for sample in SAMPLES_3_F32.iter() {
        //     f(Cast::from(*sample))
        // }
    }
}
