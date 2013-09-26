/*!

# A n-dimensional linear algebra library.

*/
#[link(name = "nalgebra"
       , vers = "0.1"
       , author = "Sébastien Crozet"
       , uuid = "1e96070f-4778-4ec1-b080-bf69f7048216")];
#[crate_type = "lib"];
#[deny(non_camel_case_types)];
#[deny(non_uppercase_statics)];
#[deny(unnecessary_qualification)];
#[deny(missing_doc)];

extern mod std;
extern mod extra;

pub mod dmat;
pub mod dvec;
pub mod vec;
pub mod mat;
pub mod types;

/// Wrappers around raw matrices to restrict their behaviour.
pub mod adaptors {
    pub mod rotmat;
    pub mod transform;
}

/// Traits implemented by matrices and vectors. Re-exported by the `mat` or the `vec` module.
pub mod traits {
    pub mod geometry;
    pub mod structure;
    pub mod operations;
}

// specialization for some 1d, 2d and 3d operations
#[doc(hidden)]
mod spec {
    mod identity;
    mod mat;
    mod vec0;
    mod vec;
}
// mod lower_triangular;
// mod chol;

#[cfg(test)]
mod tests {
    mod vec;
    mod mat;
}

#[cfg(test)]
mod bench {
    mod vec;
    mod mat;
}
