use std::mem::MaybeUninit;

// # Safety
// This trait must not be implemented outside of this crate.
pub unsafe trait InitStatus<T>: Copy {
    type Value;
    fn init(out: &mut Self::Value, t: T);
    unsafe fn assume_init_ref(t: &Self::Value) -> &T;
    unsafe fn assume_init_mut(t: &mut Self::Value) -> &mut T;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Init;
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Uninit;
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Initialized<Status>(pub Status);

unsafe impl<T> InitStatus<T> for Init {
    type Value = T;

    #[inline(always)]
    fn init(out: &mut T, t: T) {
        *out = t;
    }

    #[inline(always)]
    unsafe fn assume_init_ref(t: &T) -> &T {
        t
    }

    #[inline(always)]
    unsafe fn assume_init_mut(t: &mut T) -> &mut T {
        t
    }
}

unsafe impl<T> InitStatus<T> for Uninit {
    type Value = MaybeUninit<T>;

    #[inline(always)]
    fn init(out: &mut MaybeUninit<T>, t: T) {
        *out = MaybeUninit::new(t);
    }

    #[inline(always)]
    unsafe fn assume_init_ref(t: &MaybeUninit<T>) -> &T {
        std::mem::transmute(t.as_ptr()) // TODO: use t.assume_init_ref()
    }

    #[inline(always)]
    unsafe fn assume_init_mut(t: &mut MaybeUninit<T>) -> &mut T {
        std::mem::transmute(t.as_mut_ptr()) // TODO: use t.assume_init_mut()
    }
}

unsafe impl<T, Status: InitStatus<T>> InitStatus<T> for Initialized<Status> {
    type Value = Status::Value;

    #[inline(always)]
    fn init(out: &mut Status::Value, t: T) {
        unsafe {
            *Status::assume_init_mut(out) = t;
        }
    }

    #[inline(always)]
    unsafe fn assume_init_ref(t: &Status::Value) -> &T {
        Status::assume_init_ref(t)
    }

    #[inline(always)]
    unsafe fn assume_init_mut(t: &mut Status::Value) -> &mut T {
        Status::assume_init_mut(t)
    }
}
