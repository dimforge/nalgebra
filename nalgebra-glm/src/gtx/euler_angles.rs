use na::{Real, U3, U4};

use aliases::{Vec, Mat};

pub fn derivedEulerAngleX<N: Real>(angleX: N, angularVelocityX: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn derivedEulerAngleY<N: Real>(angleY: N, angularVelocityY: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn derivedEulerAngleZ<N: Real>(angleZ: N, angularVelocityZ: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleX<N: Real>(angleX: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleXY<N: Real>(angleX: N, angleY: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleXYX<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleXYZ<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleXZ<N: Real>(angleX: N, angleZ: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleXZX<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleXZY<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleY<N: Real>(angleY: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleYX<N: Real>(angleY: N, angleX: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleYXY<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleYXZ<N: Real>(yaw: N, pitch: N, roll: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleYZ<N: Real>(angleY: N, angleZ: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleYZX<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleYZY<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleZ<N: Real>(angleZ: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleZX<N: Real>(angle: N, angleX: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleZXY<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleZXZ<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleZY<N: Real>(angleZ: N, angleY: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleZYX<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn eulerAngleZYZ<N: Real>(t1: N, t2: N, t3: N) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn extractEulerAngleXYX<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleXYZ<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleXZX<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleXZY<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleYXY<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleYXZ<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleYZX<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleYZY<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleZXY<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleZXZ<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleZYX<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn extractEulerAngleZYZ<N: Real>(M: &Mat<N, U4, U4>) -> (N, N, N) {
    unimplemented!()
}

pub fn orientate2<N: Real>(angle: N) -> Mat<N, U3, U3> {
    unimplemented!()
}

pub fn orientate3<N: Real>(angles: Vec<N, U3>) -> Mat<N, U3, U3> {
    unimplemented!()
}

pub fn orientate4<N: Real>(angles: Vec<N, U3>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn yawPitchRoll<N: Real>(yaw: N, pitch: N, roll: N) -> Mat<N, U4, U4> {
    unimplemented!()
}
