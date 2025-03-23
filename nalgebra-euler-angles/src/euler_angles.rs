pub fn zeros(num_lines: &usize, num_columns: &usize) -> Vec<Vec<f64>> {
    let mut matrix: Vec<Vec<f64>> = Vec::new();

    for _ in 0..*num_lines {
        let mut line: Vec<f64> = Vec::new();
        for _ in 0..*num_columns {
            line.push(0.0 as f64);
        }
        matrix.push(line);
    }

    return matrix;
}

pub fn multiply(matrix1: &Vec<Vec<f64>>, matrix2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let num_lines1: usize = matrix1.len();
    let num_lines2: usize = matrix2.len();

    match (num_lines1 == 0, num_lines2 == 0) {
        (true, true) => panic!("Both matrices are empty"),
        (true, false) => panic!("The first matrix is empty"),
        (false, true) => panic!("The first matrix is empty"),
        _ => (),

    }



    let num_columns1: usize = matrix1[0].len();
    let num_columns2: usize = matrix2[0].len();

    match (num_columns1 == 0, num_columns2 == 0) {
        (true, true) => panic!("Both matrices are empty"),
        (true, false) => panic!("The first matrix is empty"),
        (false, true) => panic!("The first matrix is empty"),
        _ => (),

    }

    if num_columns1 != num_lines2 {
        panic!("The number of COLUMNS of the FIRST matrix must equal \
                the number of LINES of the SECOND matrix");
    }

    let mut result_matrix = zeros(&num_lines1, &num_columns2);

    for i in 0..num_lines1 {
        for j in 0..num_columns2 {

            result_matrix[i][j] = 0.0f64;
            for k in 0..num_lines2 {
                result_matrix[i][j] += &matrix1[i][k] * &matrix2[k][j];
            }

        }
    }

    return result_matrix;
}

pub fn rotation_x(degrees_angle: &f64) -> Vec<Vec<f64>> {
    let normalized_angle = degrees_angle % 360.0;

    let radians_angle = normalized_angle * std::f64::consts::PI / 180.0;

    let sin = radians_angle.sin();
    let cos = radians_angle.cos();

    return vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, cos, -sin],
        vec![0.0, sin, cos]
    ];
}


pub fn rotation_y(degrees_angle: &f64) -> Vec<Vec<f64>> {
    let normalized_angle = degrees_angle % 360.0;

    let radians_angle = normalized_angle * std::f64::consts::PI / 180.0;

    let sin = radians_angle.sin();
    let cos = radians_angle.cos();

    return vec![
        vec![cos, 0.0, sin],
        vec![0.0, 1.0, 0.0],
        vec![-sin, 0.0, cos]
    ];
}



pub fn rotation_z(degrees_angle: &f64) -> Vec<Vec<f64>> {
    let normalized_angle = degrees_angle % 360.0;

    let radians_angle = normalized_angle * std::f64::consts::PI / 180.0;

    let sin = radians_angle.sin();
    let cos = radians_angle.cos();

    return vec![
        vec![cos, -sin, 0.0],
        vec![sin, cos, 0.0],
        vec![0.0, 0.0, 1.0]
    ];
}


pub fn euler_to_rotation_matrix(degrees_angles: &[f64; 3], order: &[char; 3]) -> Result<Vec<Vec<f64>>, String> {
    for chr in order {
        if !matches!(*chr, 'X' | 'Y' | 'Z') {
            let err_msg = format!("Invalid order character '{}'. Expected X/Y/Z", chr);
            return Err(err_msg);
        }
    }

    let rx = rotation_x(&degrees_angles[0]);
    let ry = rotation_y(&degrees_angles[1]);
    let rz = rotation_z(&degrees_angles[2]);



    let mut result = vec![
        // Identity matrix
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0]
    ];




    for &axis in order {
        match axis {
            'X' => result = multiply(&result, &rx),
            'Y' => result = multiply(&result, &ry),
            'Z' => result = multiply(&result, &rz),
            _ => return Err(format!("Unexpected axis '{}'", axis)),
        }
    }

    Ok(result)
}



use std::collections::HashMap;


pub fn rotation_matrix_to_euler(rotation_matrix: &Vec<Vec<f64>>, order: &[char; 3]) -> Result<Vec<f64>, String> {
    // Validate the order
    for chr in order {
        if !matches!(*chr, 'X' | 'Y' | 'Z') {
            let err_msg = format!("Invalid order character '{}'. Expected X/Y/Z", chr);
            return Err(err_msg);
        }
    }

    let beta = -rotation_matrix[2][0].asin();                                                     // Y-axis rotation (beta)
    let alpha = (rotation_matrix[2][1] / beta.cos()).atan2(rotation_matrix[2][2] / beta.cos());   // X-axis rotation (alpha)
    let gamma = (rotation_matrix[1][0] / beta.cos()).atan2(rotation_matrix[0][0] / beta.cos());   // Z-axis rotation (gamma)

    // Create a map to associate axes with their corresponding Euler angle functions
    let mut euler_map: HashMap<char, f64> = HashMap::new();

    euler_map.insert('X', alpha);
    euler_map.insert('Y', beta);
    euler_map.insert('Z', gamma);

    let mut euler_angles = Vec::new();

    for &axis in order {
        match euler_map.get(&axis) {
            Some(&angle) => euler_angles.push(angle), 
            None => return Err(format!("Unexpected axis '{}'", axis)),
        }
    }

    Ok(euler_angles)
}


