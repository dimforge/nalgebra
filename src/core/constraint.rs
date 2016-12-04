use core::dimension::{Dim, DimName, Dynamic};

/// A type for enforcing constraints.
pub struct ShapeConstraint;

/// Constraints `C1` and `R2` to be equivalent.
pub trait AreMultipliable<R1: Dim, C1: Dim,
                          R2: Dim, C2: Dim> {
}


impl<R1: Dim, C1: Dim, R2: Dim, C2: Dim> AreMultipliable<R1, C1, R2, C2> for ShapeConstraint
where ShapeConstraint: DimEq<C1, R2> {
}

macro_rules! equality_trait_decl(
    ($($Trait: ident),* $(,)*) => {$(
        // XXX: we can't do something like `DimEq<D1> for D2` because we would require a blancket impl…
        pub trait $Trait<D1: Dim, D2: Dim> {
            type Representative: Dim;
        }

        impl<D: Dim> $Trait<D, D> for ShapeConstraint {
            type Representative = D;
        }

        impl<D: DimName> $Trait<D, Dynamic> for ShapeConstraint {
            type Representative = D;
        }

        impl<D: DimName> $Trait<Dynamic, D> for ShapeConstraint {
            type Representative = D;
        }
    )*}
);

equality_trait_decl!(DimEq, SameNumberOfRows, SameNumberOfColumns);

/// Constraints D1 and D2 to be equivalent, where the both designates dimensions of algebraic
/// entities (e.g. square matrices).
pub trait SameDimension<D1: Dim, D2: Dim>: SameNumberOfRows<D1, D2> + SameNumberOfColumns<D1, D2> {
    type Representative: Dim;
}

impl<D: Dim> SameDimension<D, D> for ShapeConstraint {
    type Representative = D;
}

impl<D: DimName> SameDimension<D, Dynamic> for ShapeConstraint {
    type Representative = D;
}

impl<D: DimName> SameDimension<Dynamic, D> for ShapeConstraint {
    type Representative = D;
}
