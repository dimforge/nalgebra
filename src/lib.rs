/*!

# A n-dimensional linear algebra library.

*/
#[link(name = "nalgebra"
       , vers = "0.1"
       , author = "Sébastien Crozet"
       , uuid = "1E96070F-4778-4EC1-B080-BF69F7048216")];
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

// specialization for some 1d, 2d and 3d operations
mod mat_spec;
mod vec_spec;
mod vec0_spec;
mod identity_spec;

/// Wrappers around raw matrices to restrict their behaviour.
pub mod adaptors {
    pub mod rotmat;
    pub mod transform;
}

pub mod types;

pub mod traits {
    pub mod vector;
    pub mod sample;
    pub mod indexable;
    pub mod column;
    pub mod row;
    pub mod iterable;
    pub mod outer;
    pub mod cross;
    pub mod inv;
    pub mod transpose;
    pub mod dim;
    pub mod basis;
    pub mod rotation;
    pub mod translation;
    pub mod transformation;
    pub mod rlmul;
    pub mod scalar_op;
    pub mod homogeneous;
    pub mod vec_cast;
    pub mod mat_cast;
    pub mod norm;
    pub mod dot;
    pub mod mat;
    pub mod absolute;
    pub mod col;

    /// Unusual traits which are composition of other primitive traits.
    /// Those are mainly shortcuts to make some operation easier to use or faster.
    /// Mathematics purists should really not go in there!
    pub mod comp {
        pub mod rotation_with_translation;
        pub mod absolute_rotate;
    }
}

#[cfg(test)]
mod tests {
    mod mat;
    mod vec;
}

#[cfg(test)]
mod bench {
    mod mat;
    mod vec;
}
