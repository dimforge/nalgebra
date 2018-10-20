#![cfg_attr(rustfmt, rustfmt_skip)]

use na::{Matrix4x5, CsMatrix};

#[test]
fn cs_from_to_matrix() {
    let m = Matrix4x5::new(
        5.0, 6.0, 0.0, 8.0, 15.0,
        9.0, 10.0, 11.0, 12.0, 0.0,
        0.0, 0.0, 13.0, 0.0, 0.0,
        0.0, 1.0, 4.0, 0.0, 14.0,
    );

    let cs: CsMatrix<_, _, _> = m.into();
    let m2: Matrix4x5<_> = cs.into();

    assert_eq!(m2, m);
}
