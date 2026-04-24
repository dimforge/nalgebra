use matrixcompare::assert_matrix_eq;
use nalgebra::Complex;
use nalgebra::matrix;
use nalgebra_sparse::CooMatrix;
use nalgebra_sparse::io::{
    load_coo_from_matrix_market_file, load_coo_from_matrix_market_str, save_to_matrix_market_file,
    save_to_matrix_market_str,
};
use nalgebra_sparse::proptest::coo_no_duplicates;
use proptest::prelude::*;
use tempfile::tempdir;

type C64 = Complex<f64>;
type C32 = Complex<f32>;

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_sparse_real_general_empty() {
    // Test several valid zero-shapes of a sparse matrix
    let shapes = vec![ (0, 0), (1, 0), (0, 1) ];
    let strings: Vec<String> = shapes
        .iter()
        .map(|(m, n)| format!("%%MatrixMarket matrix coordinate real general\n {} {} 0", m, n))
        .collect();

    for (shape,string) in shapes.iter().zip(strings.iter()) {
        let sparse_mat = load_coo_from_matrix_market_str::<f32>(string).unwrap();
        assert_eq!(sparse_mat.nrows(), shape.0);
        assert_eq!(sparse_mat.ncols(), shape.1);
        assert_eq!(sparse_mat.nnz(), 0);
    }
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_dense_real_general_empty() {
    // Test several valid zero-shapes of a dense matrix
    let shapes = vec![ (0, 0), (1, 0), (0, 1) ];
    let strings: Vec<String> = shapes
        .iter()
        .map(|(m, n)| format!("%%MatrixMarket matrix array real general\n {} {}", m, n))
        .collect();

    for (shape,string) in shapes.iter().zip(strings.iter()) {
        let sparse_mat = load_coo_from_matrix_market_str::<f32>(string).unwrap();
        assert_eq!(sparse_mat.nrows(), shape.0);
        assert_eq!(sparse_mat.ncols(), shape.1);
        assert_eq!(sparse_mat.nnz(), 0);
    }
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_sparse_real_general() {
    let file_str = r#"
%%MatrixMarket matrix CoOrdinate real general
% This is also an example of free-format features.
%=================================================================================
%
% This ASCII file represents a sparse MxN matrix with L
% nonzeros in the following Matrix Market format:
%
% +----------------------------------------------+
% |%%MatrixMarket matrix coordinate real general | <--- header line
% |%                                             | <--+
% |% comments                                    |    |-- 0 or more comment lines
% |%                                             | <--+
% |    M  T  L                                   | <--- rows, columns, entries
% |    I1  J1  A(I1, J1)                         | <--+
% |    I2  J2  A(I2, J2)                         |    |
% |    I3  J3  A(I3, J3)                         |    |-- L lines
% |        . . .                                 |    |
% |    IL JL  A(IL, JL)                          | <--+
% +----------------------------------------------+
%
% Indices are 1-based, i.e. A(1,1) is the first element.
%
%=================================================================================
  5  5       8
    1  			   1   	 	  1
    
    2     2     1.050e+01             
    3     3     1.500e-02   			 
		   		 	
				
   
    
  				   	 
      


    1     4             6.000e+00
                  
    4     2             2.505e+02

4     4  -2.800e+02
4     5   3.332e+01
    5     5   1.200e+01
"#;
    let sparse_mat = load_coo_from_matrix_market_str::<f32>(file_str).unwrap();
    let expected = matrix![
        1.0,   0.0,    0.0,    6.0,    0.0;
        0.0,  10.5,    0.0,    0.0,    0.0;
        0.0,   0.0,  0.015,    0.0,    0.0;
        0.0, 250.5,    0.0, -280.0,  33.32;
        0.0,   0.0,    0.0,    0.0,    12.0;
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_sparse_int_symmetric() {
    let file_str = r#"
%%MatrixMarket matrix coordinate integer symmetric
%
    5  5  9
    1  1  11
    2  2  22
    3  2  23
    3  3  33
    4  2  24
    4  4  44
    5  1  -15
    5  3  35
    5  5  55
"#;
    let sparse_mat = load_coo_from_matrix_market_str::<i128>(file_str).unwrap();
    let expected = matrix![
         11,  0,  0,   0, -15;
          0, 22, 23,  24,   0;
          0, 23, 33,   0,  35;
          0, 24,  0,  44,   0;
        -15,  0, 35,   0,  55;
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_sparse_complex_hermitian() {
    let file_str = r#"
%%MatrixMarket matrix coordinate complex hermitian
%
    5 5 7
    1 1     1.0    0.0
    2 2    10.5    0.0
    4 2   250.5   22.22
    3 3     0.015  0.0
    4 4    -2.8e2  0.0
    5 5   12.0     0.0
    5 4    0.0    33.32

"#;
    let sparse_mat = load_coo_from_matrix_market_str::<Complex<f64>>(file_str).unwrap();
    let expected = matrix![
        C64{re:1.0,im:0.0}, C64{re:0.0,im:0.0}, C64{re:0.0,im:0.0}, C64{re:0.0,im:0.0},C64{re:0.0,im:0.0};
        C64{re:0.0,im:0.0}, C64{re:10.5,im:0.0}, C64{re:0.0,im:0.0}, C64{re:250.5,im:-22.22},C64{re:0.0,im:0.0};
        C64{re:0.0,im:0.0}, C64{re:0.0,im:0.0}, C64{re:0.015,im:0.0}, C64{re:0.0,im:0.0},C64{re:0.0,im:0.0};
        C64{re:0.0,im:0.0}, C64{re:250.5,im:22.22}, C64{re:0.0,im:0.0}, C64{re:-280.0,im:0.0},C64{re:0.0,im:-33.32};
        C64{re:0.0,im:0.0}, C64{re:0.0,im:0.0}, C64{re:0.0,im:0.0}, C64{re:0.0,im:33.32},C64{re:12.0,im:0.0};
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_sparse_real_skew() {
    let file_str = r#"
%%MatrixMarket matrix coordinate real skew-symmetric
%
    5  5  4
    3  2  -23.0
    4  2  -24.0
    5  1  -15.0
    5  3  -35.0
"#;
    let sparse_mat = load_coo_from_matrix_market_str::<f64>(file_str).unwrap();
    let expected = matrix![
      0.0,    0.0,   0.0,   0.0,  15.0;
      0.0,    0.0,  23.0,  24.0,   0.0;
      0.0,  -23.0,   0.0,   0.0,  35.0;
      0.0,  -24.0,   0.0,   0.0,   0.0;
    -15.0,    0.0, -35.0,   0.0,   0.0;
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_sparse_pattern_general() {
    let file_str = r#"
%%MatrixMarket matrix coordinate pattern general
%
    5  5  10
    1  1
    1  5
    2  3
    2  4
    3  2
    3  5
    4  1
    5  2
    5  4
    5  5
"#;
    let pattern_matrix = load_coo_from_matrix_market_str::<()>(file_str).unwrap();
    let nrows = pattern_matrix.nrows();
    let ncols = pattern_matrix.ncols();
    let (row_idx, col_idx, val) = pattern_matrix.clone().disassemble();
    let values = vec![1; val.len()];
    let sparse_mat = CooMatrix::try_from_triplets(nrows, ncols, row_idx, col_idx, values).unwrap();
    let expected = matrix![
        1, 0, 0, 0, 1;
        0, 0, 1, 1, 0;
        0, 1, 0, 0, 1;
        1, 0, 0, 0, 0;
        0, 1, 0, 1, 1;
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_dense_real_general() {
    let file_str = r#"
%%MatrixMarket matrix array real general
%
4 3
1.0
2.0
3.0
4.0
5.0
6.0
7.0
8.0
9.0
10.0
11.0
12.0

"#;
    let sparse_mat = load_coo_from_matrix_market_str::<f32>(file_str).unwrap();
    let expected = matrix![
        1.0, 5.0,  9.0;
        2.0, 6.0, 10.0;
        3.0, 7.0, 11.0;
        4.0, 8.0, 12.0;
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_dense_real_symmetric() {
    let file_str = r#"
%%MatrixMarket matrix array real symmetric
%
4 4
1.0
2.0
3.0
4.0
5.0
6.0
7.0
8.0
9.0
10.0

"#;
    let sparse_mat = load_coo_from_matrix_market_str::<f32>(file_str).unwrap();
    let expected = matrix![
        1.0, 2.0, 3.0,  4.0;
        2.0, 5.0, 6.0,  7.0;
        3.0, 6.0, 8.0,  9.0;
        4.0, 7.0, 9.0, 10.0;
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_dense_complex_hermitian() {
    let file_str = r#"
%%MatrixMarket matrix array complex hermitian
%
4 4
1.0 0.0
2.0 2.0
3.0 3.0
4.0 4.0
5.0 0.0
6.0 6.0
7.0 7.0
8.0 0.0
9.0 9.0
10.0 0.0

"#;
    let sparse_mat = load_coo_from_matrix_market_str::<C64>(file_str).unwrap();
    let expected = matrix![
        C64{re:1.0,im:0.0}, C64{re:2.0,im:-2.0} ,C64{re:3.0,im:-3.0} ,C64{re:4.0,im:-4.0};
        C64{re:2.0,im:2.0}, C64{re:5.0,im:0.0} ,C64{re:6.0,im:-6.0} ,C64{re:7.0,im:-7.0};
        C64{re:3.0,im:3.0}, C64{re:6.0,im:6.0} ,C64{re:8.0,im:0.0} ,C64{re:9.0,im:-9.0};
        C64{re:4.0,im:4.0}, C64{re:7.0,im:7.0} ,C64{re:9.0,im:9.0} ,C64{re:10.0,im:0.0};
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_dense_int_skew() {
    let file_str = r#"
%%MatrixMarket matrix array integer skew-symmetric
%
4 4
1
2
3
4
5
6
"#;
    let sparse_mat = load_coo_from_matrix_market_str::<i32>(file_str).unwrap();
    let expected = matrix![
        0,-1,-2,-3;
        1, 0,-4,-5;
        2, 4, 0,-6;
        3, 5, 6, 0;
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_load_dense_complex_general() {
    let file_str = r#"
%%MatrixMarket matrix array complex general
%
2 2
1 0
1 0
1 0
1 0
"#;
    let sparse_mat = load_coo_from_matrix_market_str::<C32>(file_str).unwrap();
    let expected = matrix![
        C32{re:1.0,im:0.0},C32{re:1.0,im:0.0};
        C32{re:1.0,im:0.0},C32{re:1.0,im:0.0};
    ];
    assert_matrix_eq!(sparse_mat, expected);
}

#[test]
#[rustfmt::skip]
fn test_matrixmarket_write_real(){
    let dense_matrix = matrix![
        1.0, 2.0, 3.0;
        2.0, 0.0, 3.0;
        ];
    let row_indices = vec![0,1,0,0,1];
    let col_indices = vec![0,0,1,2,2];
    let values = vec![1.0,2.0,2.0,3.0,3.0];
    let coo_matrix = CooMatrix::try_from_triplets(2, 3, row_indices, col_indices, values).unwrap();
    assert_matrix_eq!(dense_matrix,coo_matrix);
    let expected = r#"%%matrixmarket matrix coordinate real general
% matrixmarket file generated by nalgebra-sparse.
2 3 5
1 1 1
2 1 2
1 2 2
1 3 3
2 3 3
"#;
    let matrixmarket_str = save_to_matrix_market_str(&coo_matrix);
    assert_eq!(matrixmarket_str,expected);
}

#[test]
fn test_matrixmarket_write_int() {
    let dense_matrix = matrix![
    1,2,3;
    2,0,3;
    ];
    let row_indices = vec![0, 1, 0, 0, 1];
    let col_indices = vec![0, 0, 1, 2, 2];
    let values = vec![1, 2, 2, 3, 3];
    let coo_matrix = CooMatrix::try_from_triplets(2, 3, row_indices, col_indices, values).unwrap();
    assert_matrix_eq!(dense_matrix, coo_matrix);
    let expected = r#"%%matrixmarket matrix coordinate integer general
% matrixmarket file generated by nalgebra-sparse.
2 3 5
1 1 1
2 1 2
1 2 2
1 3 3
2 3 3
"#;
    let matrixmarket_str = save_to_matrix_market_str(&coo_matrix);
    assert_eq!(matrixmarket_str, expected);
}

#[test]
fn test_matrixmarket_write_pattern() {
    let row_indices = vec![0, 1, 0, 0, 1];
    let col_indices = vec![0, 0, 1, 2, 2];
    let values = vec![(), (), (), (), ()];
    let coo_matrix = CooMatrix::try_from_triplets(2, 3, row_indices, col_indices, values).unwrap();
    let expected = r#"%%matrixmarket matrix coordinate pattern general
% matrixmarket file generated by nalgebra-sparse.
2 3 5
1 1 
2 1 
1 2 
1 3 
2 3 
"#;
    let matrixmarket_str = save_to_matrix_market_str(&coo_matrix);
    assert_eq!(matrixmarket_str, expected);
}

#[test]
fn test_matrixmarket_write_complex() {
    let row_indices = vec![0, 1, 0, 0, 1];
    let col_indices = vec![0, 0, 1, 2, 2];
    let values = vec![
        C64 { re: 1.0, im: 2.0 },
        C64 { re: 2.0, im: 3.0 },
        C64 { re: 3.0, im: 4.0 },
        C64 { re: 4.0, im: 5.0 },
        C64 { re: 5.0, im: 6.0 },
    ];
    let coo_matrix = CooMatrix::try_from_triplets(2, 3, row_indices, col_indices, values).unwrap();
    let expected = r#"%%matrixmarket matrix coordinate complex general
% matrixmarket file generated by nalgebra-sparse.
2 3 5
1 1 1 2
2 1 2 3
1 2 3 4
1 3 4 5
2 3 5 6
"#;
    let matrixmarket_str = save_to_matrix_market_str(&coo_matrix);
    assert_eq!(matrixmarket_str, expected);
}

proptest! {
    #[test]
    fn coo_matrix_market_roundtrip_str(coo in coo_no_duplicates(-10 ..= 10, 0 ..= 10, 0..= 10, 100)) {
        let generated_matrixmarket_string = save_to_matrix_market_str(&coo);
        let generated_matrix = load_coo_from_matrix_market_str(&generated_matrixmarket_string).unwrap();
        assert_matrix_eq!(generated_matrix, coo);
    }
}

proptest! {
    #[test]
    fn coo_matrix_market_roundtrip_file(coo in coo_no_duplicates(-10 ..= 10, 0 ..= 10, 0..= 10, 100)) {
        let temp_dir = tempdir().expect("Unable to create temporary directory");
        let file_path = temp_dir.path().join("temp.mtx");
        save_to_matrix_market_file(&coo,&file_path).unwrap();
        let generated_matrix = load_coo_from_matrix_market_file(file_path).unwrap();
        assert_matrix_eq!(generated_matrix, coo);
        temp_dir.close().expect("Unable to delete temporary directory");
    }
}
