//! The sparse matrix operations must never drop or overwrite uninitialized values; this is
//! observable with a scalar type that owns heap memory (see #1610).
#![cfg(feature = "sparse")]

use nalgebra::{CsMatrix, DMatrix};
use num_traits::{One, Zero};
use std::cell::Cell;
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::panic::{AssertUnwindSafe, catch_unwind};

thread_local! {
    static PANIC_ON_CLONE: Cell<bool> = const { Cell::new(false) };
}

#[derive(Debug, PartialEq)]
struct Boxed(Box<f64>);

impl Clone for Boxed {
    fn clone(&self) -> Self {
        if PANIC_ON_CLONE.with(Cell::get) {
            panic!("clone panicked");
        }
        Boxed(self.0.clone())
    }
}

fn b(x: f64) -> Boxed {
    Boxed(Box::new(x))
}

impl Add for Boxed {
    type Output = Boxed;
    fn add(self, rhs: Boxed) -> Boxed {
        b(*self.0 + *rhs.0)
    }
}
impl AddAssign for Boxed {
    fn add_assign(&mut self, rhs: Boxed) {
        *self.0 += *rhs.0;
    }
}
impl Mul for Boxed {
    type Output = Boxed;
    fn mul(self, rhs: Boxed) -> Boxed {
        b(*self.0 * *rhs.0)
    }
}
impl MulAssign for Boxed {
    fn mul_assign(&mut self, rhs: Boxed) {
        *self.0 *= *rhs.0;
    }
}
impl Zero for Boxed {
    fn zero() -> Self {
        b(0.0)
    }
    fn is_zero(&self) -> bool {
        *self.0 == 0.0
    }
}
impl One for Boxed {
    fn one() -> Self {
        b(1.0)
    }
}

fn dense() -> DMatrix<Boxed> {
    DMatrix::from_fn(3, 4, |i, j| {
        if (i + j) % 2 == 0 {
            b((i * 4 + j + 1) as f64)
        } else {
            b(0.0)
        }
    })
}

#[test]
fn cs_heap_scalar_conversion_transpose_and_ops() {
    let m = dense();
    let cs: CsMatrix<Boxed> = m.clone().into();
    assert_eq!(DMatrix::from(cs.clone()), m);

    let t = cs.transpose();
    assert_eq!(DMatrix::from(t.clone()), m.transpose());

    let sum = &cs + &cs;
    assert_eq!(DMatrix::from(sum), m.map(|x| b(*x.0 * 2.0)));

    let prod = &cs * &t;
    let expected = DMatrix::from_fn(3, 3, |i, j| {
        b((0..4).map(|k| *m[(i, k)].0 * *m[(j, k)].0).sum())
    });
    assert_eq!(DMatrix::from(prod), expected);

    let trip = CsMatrix::from_triplet(2, 2, &[0, 1, 1], &[0, 0, 1], &[b(1.0), b(2.0), b(3.0)]);
    let expected = DMatrix::from_row_slice(2, 2, &[b(1.0), b(0.0), b(2.0), b(3.0)]);
    assert_eq!(DMatrix::from(trip), expected);
}

#[test]
fn cs_transpose_panicking_clone_is_safe() {
    let cs: CsMatrix<Boxed> = dense().into();
    PANIC_ON_CLONE.with(|p| p.set(true));
    let res = catch_unwind(AssertUnwindSafe(|| cs.transpose()));
    PANIC_ON_CLONE.with(|p| p.set(false));
    assert!(res.is_err());
    // `cs` is still valid and dropping it (and the unwound partial result) is sound.
    assert_eq!(DMatrix::from(cs), dense());
}
