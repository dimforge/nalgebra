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
primitive_double_dispatch_cast_decl_trait!(int,  intCast);
primitive_double_dispatch_cast_decl_trait!(uint, uintCast);

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
primitive_cast_redispatch_impl!(int,  intCast);
primitive_cast_redispatch_impl!(uint, uintCast);

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
primitive_double_dispatch_cast_impl!(f64, int,  f64Cast);
primitive_double_dispatch_cast_impl!(f64, uint, f64Cast);

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
primitive_double_dispatch_cast_impl!(f32, int,  f32Cast);
primitive_double_dispatch_cast_impl!(f32, uint, f32Cast);

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
primitive_double_dispatch_cast_impl!(i64, int,  i64Cast);
primitive_double_dispatch_cast_impl!(i64, uint, i64Cast);

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
primitive_double_dispatch_cast_impl!(i32, int,  i32Cast);
primitive_double_dispatch_cast_impl!(i32, uint, i32Cast);

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
primitive_double_dispatch_cast_impl!(i16, int,  i16Cast);
primitive_double_dispatch_cast_impl!(i16, uint, i16Cast);

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
primitive_double_dispatch_cast_impl!(i8, int,  i8Cast);
primitive_double_dispatch_cast_impl!(i8, uint, i8Cast);

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
primitive_double_dispatch_cast_impl!(u64, int,  u64Cast);
primitive_double_dispatch_cast_impl!(u64, uint, u64Cast);

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
primitive_double_dispatch_cast_impl!(u32, int,  u32Cast);
primitive_double_dispatch_cast_impl!(u32, uint, u32Cast);

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
primitive_double_dispatch_cast_impl!(u16, int,  u16Cast);
primitive_double_dispatch_cast_impl!(u16, uint, u16Cast);

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
primitive_double_dispatch_cast_impl!(u8, int,  u8Cast);
primitive_double_dispatch_cast_impl!(u8, uint, u8Cast);

primitive_double_dispatch_cast_impl!(uint, f64,  uintCast);
primitive_double_dispatch_cast_impl!(uint, f32,  uintCast);
primitive_double_dispatch_cast_impl!(uint, i64,  uintCast);
primitive_double_dispatch_cast_impl!(uint, i32,  uintCast);
primitive_double_dispatch_cast_impl!(uint, i16,  uintCast);
primitive_double_dispatch_cast_impl!(uint, i8,   uintCast);
primitive_double_dispatch_cast_impl!(uint, u64,  uintCast);
primitive_double_dispatch_cast_impl!(uint, u32,  uintCast);
primitive_double_dispatch_cast_impl!(uint, u16,  uintCast);
primitive_double_dispatch_cast_impl!(uint, u8,   uintCast);
primitive_double_dispatch_cast_impl!(uint, int,  uintCast);
primitive_double_dispatch_cast_impl!(uint, uint, uintCast);

primitive_double_dispatch_cast_impl!(int, f64,  intCast);
primitive_double_dispatch_cast_impl!(int, f32,  intCast);
primitive_double_dispatch_cast_impl!(int, i64,  intCast);
primitive_double_dispatch_cast_impl!(int, i32,  intCast);
primitive_double_dispatch_cast_impl!(int, i16,  intCast);
primitive_double_dispatch_cast_impl!(int, i8,   intCast);
primitive_double_dispatch_cast_impl!(int, u64,  intCast);
primitive_double_dispatch_cast_impl!(int, u32,  intCast);
primitive_double_dispatch_cast_impl!(int, u16,  intCast);
primitive_double_dispatch_cast_impl!(int, u8,   intCast);
primitive_double_dispatch_cast_impl!(int, int,  intCast);
primitive_double_dispatch_cast_impl!(int, uint, intCast);
