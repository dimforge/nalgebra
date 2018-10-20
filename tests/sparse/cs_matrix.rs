#![cfg_attr(rustfmt, rustfmt_skip)]

use na::{Matrix4x5, Matrix5x4, CsMatrix};

#[test]
fn cs_transpose() {
    let m = Matrix4x5::new(
        4.0, 1.0, 4.0, 0.0, 9.0,
        5.0, 6.0, 0.0, 8.0, 10.0,
        9.0, 10.0, 11.0, 12.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 10.0
    );

    let cs: CsMatrix<_, _, _> = m.into();
    let cs_transposed: Matrix5x4<_> = cs.transpose().into();

    assert_eq!(cs_transposed, m.transpose())
}
