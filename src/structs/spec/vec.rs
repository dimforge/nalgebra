use std::num::{Zero, One};
use traits::structure::{Cast, Row, Basis};
use traits::geometry::{Norm, Cross, CrossMatrix, UniformSphereSample};
use structs::vec::{Vec1, Vec2, Vec3, Vec4};
use structs::mat::Mat3;

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec1<N>> for Vec2<N> {
    #[inline]
    fn cross(a: &Vec2<N>, b: &Vec2<N>) -> Vec1<N> {
        Vec1::new(a.x * b.y - a.y * b.x)
    }
}

// FIXME: instead of returning a Vec2, define a Mat2x1 matrix?
impl<N: Neg<N> + Clone> CrossMatrix<Vec2<N>> for Vec2<N> {
    #[inline]
    fn cross_matrix(v: &Vec2<N>) -> Vec2<N> {
        Vec2::new(-v.y, v.x.clone())
    }
}

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec3<N>> for Vec3<N> {
    #[inline]
    fn cross(a: &Vec3<N>, b: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        )
    }
}

impl<N: Neg<N> + Zero + Clone> CrossMatrix<Mat3<N>> for Vec3<N> {
    #[inline]
    fn cross_matrix(v: &Vec3<N>) -> Mat3<N> {
        Mat3::new(
            Zero::zero(), -v.z        , v.y.clone(),
            v.z.clone() , Zero::zero(), -v.x,
            -v.y        , v.x.clone() , Zero::zero()
        )
    }
}

// FIXME: implement this for all other vectors
impl<N: Clone> Row<Vec1<N>> for Vec2<N> {
    #[inline]
    fn nrows(&self) -> uint {
        2
    }

    #[inline]
    fn row(&self, i: uint) -> Vec1<N> {
        match i {
            0 => Vec1::new(self.x.clone()),
            1 => Vec1::new(self.y.clone()),
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
        f(Vec1::new(One::one()));
    }

    #[inline(always)]
    fn orthonormal_subspace_basis(_: &Vec1<N>, _: |Vec1<N>| -> bool ) { }
}

impl<N: Clone + One + Zero + Neg<N>> Basis for Vec2<N> {
    #[inline(always)]
    fn canonical_basis(f: |Vec2<N>| -> bool) {
        if !f(Vec2::new(One::one(), Zero::zero())) { return };
        f(Vec2::new(Zero::zero(), One::one()));
    }

    #[inline]
    fn orthonormal_subspace_basis(n: &Vec2<N>, f: |Vec2<N>| -> bool) {
        f(Vec2::new(-n.y, n.x.clone()));
    }
}

impl<N: Float> Basis for Vec3<N> {
    #[inline(always)]
    fn canonical_basis(f: |Vec3<N>| -> bool) {
        if !f(Vec3::new(One::one(), Zero::zero(), Zero::zero())) { return };
        if !f(Vec3::new(Zero::zero(), One::one(), Zero::zero())) { return };
        f(Vec3::new(Zero::zero(), Zero::zero(), One::one()));
    }

    #[inline(always)]
    fn orthonormal_subspace_basis(n: &Vec3<N>, f: |Vec3<N>| -> bool) {
        let a = 
            if n.x.clone().abs() > n.y.clone().abs() {
                Norm::normalize_cpy(&Vec3::new(n.z.clone(), Zero::zero(), -n.x))
            }
            else {
                Norm::normalize_cpy(&Vec3::new(Zero::zero(), -n.z, n.y.clone()))
            };

        if !f(Cross::cross(&a, n)) { return };
        f(a);
    }
}

// FIXME: this bad: this fixes definitly the number of samples…
static SAMPLES_2_F32: [Vec2<f32>, ..21] = [
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
static SAMPLES_3_F32: [Vec3<f32>, ..42] = [
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
    fn sample(f: |Vec1<N>| -> ()) {
        f(One::one())
     }
}

impl<N: Cast<f32> + Clone> UniformSphereSample for Vec2<N> {
    #[inline(always)]
    fn sample(f: |Vec2<N>| -> ()) {
         for sample in SAMPLES_2_F32.iter() {
             f(Cast::from(*sample))
         }
     }
}

impl<N: Cast<f32> + Clone> UniformSphereSample for Vec3<N> {
    #[inline(always)]
    fn sample(f: |Vec3<N>| -> ()) {
        for sample in SAMPLES_3_F32.iter() {
            f(Cast::from(*sample))
        }
    }
}

impl<N: Cast<f32> + Clone> UniformSphereSample for Vec4<N> {
    #[inline(always)]
    fn sample(_: |Vec4<N>| -> ()) {
        panic!("UniformSphereSample::<Vec4<N>>::sample : Not yet implemented.")
        // for sample in SAMPLES_3_F32.iter() {
        //     f(Cast::from(*sample))
        // }
    }
}
