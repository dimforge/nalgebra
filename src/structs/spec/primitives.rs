//! nalgebra trait implementation for primitive types.

#![allow(missing_docs)]
#![allow(non_camel_case_types)]

use traits::structure::Cast;

// Double dispatch traits to drive the Cast method for primitive types.
macro_rules! primitive_double_dispatch_cast_decl_trait(
    ($t: ident, $tcast: ident) => (
        pub trait $tcast {
            fn to(Self) -> $t;
        }
    )
);

macro_rules! primitive_double_dispatch_cast_impl(
    ($ttarget: ident, $tself: ident, $tcast: ident) => (
        impl $tcast for $tself{
            #[inline(always)]
            fn to(v: $tself) -> $ttarget {
                v as $ttarget
            }
        }
    )
);

macro_rules! primitive_cast_redispatch_impl(
    ($t:ident, $tcast: ident) => (
        impl<T: $tcast> Cast<T> for $t {
            #[inline(always)]
            fn from(t: T) -> $t {
                $tcast::to(t)
            }
        }
    )
);

primitive_double_dispatch_cast_decl_trait!(f64,  f64Cast);
primitive_double_dispatch_cast_decl_trait!(f32,  f32Cast);
primitive_double_dispatch_cast_decl_trait!(i64,  i64Cast);
primitive_double_dispatch_cast_decl_trait!(i32,  i32Cast);
primitive_double_dispatch_cast_decl_trait!(i16,  i16Cast);
primitive_double_dispatch_cast_decl_trait!(i8,   i8Cast);
primitive_double_dispatch_cast_decl_trait!(u64,  u64Cast);
primitive_double_dispatch_cast_decl_trait!(u32,  u32Cast);
primitive_double_dispatch_cast_decl_trait!(u16,  u16Cast);
primitive_double_dispatch_cast_decl_trait!(u8,   u8Cast);
primitive_double_dispatch_cast_decl_trait!(isize,  isizeCast);
primitive_double_dispatch_cast_decl_trait!(usize, usizeCast);

primitive_cast_redispatch_impl!(f64,  f64Cast);
primitive_cast_redispatch_impl!(f32,  f32Cast);
primitive_cast_redispatch_impl!(i64,  i64Cast);
primitive_cast_redispatch_impl!(i32,  i32Cast);
primitive_cast_redispatch_impl!(i16,  i16Cast);
primitive_cast_redispatch_impl!(i8,   i8Cast);
primitive_cast_redispatch_impl!(u64,  u64Cast);
primitive_cast_redispatch_impl!(u32,  u32Cast);
primitive_cast_redispatch_impl!(u16,  u16Cast);
primitive_cast_redispatch_impl!(u8,   u8Cast);
primitive_cast_redispatch_impl!(isize,  isizeCast);
primitive_cast_redispatch_impl!(usize, usizeCast);

primitive_double_dispatch_cast_impl!(f64, f64,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, f32,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, i64,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, i32,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, i16,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, i8,   f64Cast);
primitive_double_dispatch_cast_impl!(f64, u64,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, u32,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, u16,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, u8,   f64Cast);
primitive_double_dispatch_cast_impl!(f64, isize,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, usize, f64Cast);

primitive_double_dispatch_cast_impl!(f32, f64,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, f32,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, i64,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, i32,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, i16,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, i8,   f32Cast);
primitive_double_dispatch_cast_impl!(f32, u64,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, u32,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, u16,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, u8,   f32Cast);
primitive_double_dispatch_cast_impl!(f32, isize,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, usize, f32Cast);

primitive_double_dispatch_cast_impl!(i64, f64,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, f32,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, i64,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, i32,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, i16,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, i8,   i64Cast);
primitive_double_dispatch_cast_impl!(i64, u64,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, u32,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, u16,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, u8,   i64Cast);
primitive_double_dispatch_cast_impl!(i64, isize,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, usize, i64Cast);

primitive_double_dispatch_cast_impl!(i32, f64,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, f32,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, i64,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, i32,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, i16,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, i8,   i32Cast);
primitive_double_dispatch_cast_impl!(i32, u64,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, u32,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, u16,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, u8,   i32Cast);
primitive_double_dispatch_cast_impl!(i32, isize,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, usize, i32Cast);

primitive_double_dispatch_cast_impl!(i16, f64,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, f32,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, i64,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, i32,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, i16,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, i8,   i16Cast);
primitive_double_dispatch_cast_impl!(i16, u64,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, u32,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, u16,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, u8,   i16Cast);
primitive_double_dispatch_cast_impl!(i16, isize,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, usize, i16Cast);

primitive_double_dispatch_cast_impl!(i8, f64,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, f32,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, i64,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, i32,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, i16,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, i8,   i8Cast);
primitive_double_dispatch_cast_impl!(i8, u64,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, u32,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, u16,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, u8,   i8Cast);
primitive_double_dispatch_cast_impl!(i8, isize,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, usize, i8Cast);

primitive_double_dispatch_cast_impl!(u64, f64,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, f32,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, i64,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, i32,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, i16,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, i8,   u64Cast);
primitive_double_dispatch_cast_impl!(u64, u64,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, u32,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, u16,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, u8,   u64Cast);
primitive_double_dispatch_cast_impl!(u64, isize,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, usize, u64Cast);

primitive_double_dispatch_cast_impl!(u32, f64,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, f32,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, i64,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, i32,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, i16,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, i8,   u32Cast);
primitive_double_dispatch_cast_impl!(u32, u64,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, u32,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, u16,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, u8,   u32Cast);
primitive_double_dispatch_cast_impl!(u32, isize,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, usize, u32Cast);

primitive_double_dispatch_cast_impl!(u16, f64,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, f32,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, i64,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, i32,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, i16,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, i8,   u16Cast);
primitive_double_dispatch_cast_impl!(u16, u64,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, u32,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, u16,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, u8,   u16Cast);
primitive_double_dispatch_cast_impl!(u16, isize,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, usize, u16Cast);

primitive_double_dispatch_cast_impl!(u8, f64,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, f32,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, i64,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, i32,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, i16,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, i8,   u8Cast);
primitive_double_dispatch_cast_impl!(u8, u64,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, u32,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, u16,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, u8,   u8Cast);
primitive_double_dispatch_cast_impl!(u8, isize,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, usize, u8Cast);

primitive_double_dispatch_cast_impl!(usize, f64,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, f32,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, i64,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, i32,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, i16,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, i8,   usizeCast);
primitive_double_dispatch_cast_impl!(usize, u64,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, u32,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, u16,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, u8,   usizeCast);
primitive_double_dispatch_cast_impl!(usize, isize,  usizeCast);
primitive_double_dispatch_cast_impl!(usize, usize, usizeCast);

primitive_double_dispatch_cast_impl!(isize, f64,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, f32,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, i64,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, i32,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, i16,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, i8,   isizeCast);
primitive_double_dispatch_cast_impl!(isize, u64,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, u32,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, u16,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, u8,   isizeCast);
primitive_double_dispatch_cast_impl!(isize, isize,  isizeCast);
primitive_double_dispatch_cast_impl!(isize, usize, isizeCast);
