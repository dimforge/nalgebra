use traits::ring::Ring;

pub trait DivisionRing : Ring + Quot<Self, Self>
{ }
