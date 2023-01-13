//! Wrapper that allows changing the generic type of a PhantomData<T>
//!
//! Copied from https://github.com/rkyv/rkyv_contrib (MIT-Apache2 licences) which isn’t published yet.

use rkyv::{
    with::{ArchiveWith, DeserializeWith, SerializeWith},
    Fallible,
};
use std::marker::PhantomData;

/// A wrapper that allows for changing the generic type of a PhantomData<T>.
///
/// Example:
///
/// ```rust
/// use std::marker::PhantomData;
/// use rkyv::{
///     Archive, Serialize, Deserialize, Infallible, vec::ArchivedVec, Archived, with::With,
/// };
/// use rkyv_wrappers::custom_phantom::CustomPhantom;
/// #[derive(Archive, Serialize, Deserialize, Debug, PartialEq, Eq, Default)]
/// #[archive(as = "StructWithPhantom<T::Archived>", bound(archive = "
/// 	T: Archive,
/// 	With<PhantomData<T>, CustomPhantom<Archived<T>>>: Archive<Archived = PhantomData<Archived<T>>>
/// "))]
/// struct StructWithPhantom<T> {
/// 	pub num: i32,
///     #[with(CustomPhantom<T::Archived>)]
///     pub phantom: PhantomData<T>,
/// }
/// let value = StructWithPhantom::<Vec<i32>>::default();
/// let bytes = rkyv::to_bytes::<_, 1024>(&value).unwrap();
/// let archived: &StructWithPhantom<ArchivedVec<i32>> = unsafe { rkyv::archived_root::<StructWithPhantom<Vec<i32>>>(&bytes) };
///
/// let deserialized: StructWithPhantom<Vec<i32>> = archived.deserialize(&mut Infallible).unwrap();
/// assert_eq!(deserialized, value);
/// ```
pub struct CustomPhantom<NT: ?Sized> {
    _data: PhantomData<*const NT>,
}

impl<OT: ?Sized, NT: ?Sized> ArchiveWith<PhantomData<OT>> for CustomPhantom<NT> {
    type Archived = PhantomData<NT>;
    type Resolver = ();

    #[inline]
    unsafe fn resolve_with(
        _: &PhantomData<OT>,
        _: usize,
        _: Self::Resolver,
        _: *mut Self::Archived,
    ) {
    }
}

impl<OT: ?Sized, NT: ?Sized, S: Fallible + ?Sized> SerializeWith<PhantomData<OT>, S>
    for CustomPhantom<NT>
{
    #[inline]
    fn serialize_with(_: &PhantomData<OT>, _: &mut S) -> Result<Self::Resolver, S::Error> {
        Ok(())
    }
}

impl<OT: ?Sized, NT: ?Sized, D: Fallible + ?Sized>
    DeserializeWith<PhantomData<NT>, PhantomData<OT>, D> for CustomPhantom<NT>
{
    #[inline]
    fn deserialize_with(_: &PhantomData<NT>, _: &mut D) -> Result<PhantomData<OT>, D::Error> {
        Ok(PhantomData)
    }
}
