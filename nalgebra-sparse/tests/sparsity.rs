use nalgebra_sparse::pattern::{SparsityPattern, SparsityPatternBuilder};

#[test]
fn sparsity_identity() {
    let n = 100;
    let speye = SparsityPattern::identity(n);
    for (i, j) in speye.entries() {
        assert_eq!(i, j);
    }
    assert_eq!(speye.major_dim(), n);
    assert_eq!(speye.minor_dim(), n);
}

#[test]
fn lower_sparse_solve() {
    // just a smaller number so it's easier to debug
    let n = 8;
    let speye = SparsityPattern::identity(n);
    let mut buf = vec![];
    speye.sparse_lower_triangular_solve(&[0, 5], &mut buf);
    assert_eq!(buf, vec![0, 5]);

    // test case from
    // https://www.youtube.com/watch?v=1dGRTOwBkQs&ab_channel=TimDavis
    let mut builder = SparsityPatternBuilder::new(14, 14);
    // building CscMatrix, so it will be col, row
    #[rustfmt::skip]
    let indices = vec![
      (0, 0), (0, 2),
      (1, 1), (1, 3), (1, 6), (1, 8),
      (2,2), (2,4), (2,7),
      (3,3), (3,8),
      (4,4), (4,7),
      (5,5), (5,8), (5,9),
      (6,6), (6,9), (6,10),
      (7,7), (7,9),
      (8,8), (8,11), (8,12),
      (9,9), (9,10), (9, 12), (9, 13),
      (10,10), (10,11), (10,12),
      (11,11), (11,12),
      (12,12), (12,13),
      (13,13),
    ];
    for (maj, min) in indices.iter().copied() {
        assert!(builder.insert(maj, min).is_ok());
    }
    let sp = builder.build();
    assert_eq!(sp.major_dim(), 14);
    assert_eq!(sp.minor_dim(), 14);
    assert_eq!(sp.nnz(), indices.len());
    for ij in sp.entries() {
        assert!(indices.contains(&ij));
    }
    sp.sparse_lower_triangular_solve(&[3, 5], &mut buf);
    assert_eq!(buf, vec![3, 8, 11, 12, 13, 5, 9, 10]);
}

// this test is a flipped version of lower sparse solve
#[test]
fn upper_sparse_solve() {
    // just a smaller number so it's easier to debug
    let n = 8;
    let speye = SparsityPattern::identity(n);
    let mut buf = vec![];
    speye.sparse_lower_triangular_solve(&[0, 5], &mut buf);
    assert_eq!(buf, vec![0, 5]);

    // test case from
    // https://www.youtube.com/watch?v=1dGRTOwBkQs&ab_channel=TimDavis
    let mut builder = SparsityPatternBuilder::new(14, 14);
    // building CscMatrix, so it will be col, row
    #[rustfmt::skip]
    let mut indices = vec![
      (0, 0), (0, 2),
      (1, 1), (1, 3), (1, 6), (1, 8),
      (2,2), (2,4), (2,7),
      (3,3), (3,8),
      (4,4), (4,7),
      (5,5), (5,8), (5,9),
      (6,6), (6,9), (6,10),
      (7,7), (7,9),
      (8,8), (8,11), (8,12),
      (9,9), (9,10), (9, 12), (9, 13),
      (10,10), (10,11), (10,12),
      (11,11), (11,12),
      (12,12), (12,13),
      (13,13),
    ];
    indices.sort_by_key(|&(min, maj)| (maj, min));
    for (min, maj) in indices.iter().copied() {
        assert!(builder.insert(maj, min).is_ok());
    }
    let sp = builder.build();
    assert_eq!(sp.major_dim(), 14);
    assert_eq!(sp.minor_dim(), 14);
    assert_eq!(sp.nnz(), indices.len());
    sp.sparse_upper_triangular_solve(&[9], &mut buf);
    assert_eq!(buf, vec![9, 7, 4, 2, 0, 6, 1, 5]);
}

#[test]
fn test_builder() {
    let mut builder = SparsityPatternBuilder::new(2, 2);
    assert!(builder.insert(0, 0).is_ok());
    assert!(builder.insert(0, 0).is_err());
    assert!(builder.insert(0, 1).is_ok());
    assert!(builder.insert(0, 1).is_err());
    assert!(builder.insert(1, 0).is_ok());
    assert!(builder.insert(1, 0).is_err());
}

#[test]
fn test_builder_reset() {
    let mut builder = SparsityPatternBuilder::new(4, 4);
    for i in 0..3 {
        assert!(builder.insert(i, i + 1).is_ok());
    }
    let out = builder.build();
    assert_eq!(out.major_dim(), 4);
    assert_eq!(out.minor_dim(), 4);
    let mut builder = SparsityPatternBuilder::from(out);
    for i in (0..=2).rev() {
        assert!(builder.revert_to_major(i));
        assert_eq!(builder.current_major(), i);
    }
    let out = builder.build();

    let mut builder = SparsityPatternBuilder::from(out);
    assert!(builder.revert_to_major(1));
    assert_eq!(builder.current_major(), 1);
}
