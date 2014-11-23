//! Assertion macro tests

#![feature(phase)]

#[phase(plugin)]
extern crate nalgebra;
extern crate nalgebra;

use nalgebra::{ApproxEq, Vec2};

#[test]
fn assert_approx_eq_f64() {
    let a = 1.0f64;
    let b = 1.0f64 + 1.0e-12f64;
    assert_approx_eq!(a, b);
}

#[test]
#[should_fail]
fn assert_approx_eq_vec2_f32_fail() {
    let a = Vec2::new(1.0f32, 0.0);
    let b = Vec2::new(1.1f32, 0.1);
    assert_approx_eq!(a, b);
}

#[test]
fn assert_approx_eq_eps_f32() {
    assert_approx_eq_eps!(1.0f32, 1.1, 0.2);
}

#[test]
#[should_fail]
fn assert_approx_eq_eps_f64_fail() {
    assert_approx_eq_eps!(1.0f64, 1.1, 0.05);
}
