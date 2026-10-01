pub mod euler_angles;

#[cfg(test)]
mod tests {
    use super::euler_angles::*;

    #[test]
    fn test_zeros() {
        let matrix = zeros(&2, &3);
        assert_eq!(matrix, vec![vec![0.0; 3]; 2]);
    }

    #[test]
    fn test_multiply_identity() {
        let identity = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        assert_eq!(multiply(&identity, &matrix), matrix);
    }

    #[test]
    fn test_rotation_x() {
        let rx = rotation_x(&90.0);
        assert!((rx[1][1] - 0.0).abs() < 1e-6);
        assert!((rx[1][2] - -1.0).abs() < 1e-6);
        assert!((rx[2][1] - 1.0).abs() < 1e-6);
        assert!((rx[2][2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_y() {
        let ry = rotation_y(&90.0);
        assert!((ry[0][0] - 0.0).abs() < 1e-6);
        assert!((ry[0][2] - 1.0).abs() < 1e-6);
        assert!((ry[2][0] - -1.0).abs() < 1e-6);
        assert!((ry[2][2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_z() {
        let rz = rotation_z(&90.0);
        assert!((rz[0][0] - 0.0).abs() < 1e-6);
        assert!((rz[0][1] - -1.0).abs() < 1e-6);
        assert!((rz[1][0] - 1.0).abs() < 1e-6);
        assert!((rz[1][1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euler_to_rotation_matrix() {
        let angles = [90.0, 0.0, 0.0];
        let order = ['X', 'Y', 'Z'];
        let result = euler_to_rotation_matrix(&angles, &order);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rotation_matrix_to_euler() {
        let rotation_matrix = vec![
            vec![0.8660254, -0.5, 0.0], 
            vec![0.5, 0.8660254, 0.0], 
            vec![0.0, 0.0, 1.0]
        ]; 

        let order = ['Z', 'Y', 'X']; // Rotation order

        let result = rotation_matrix_to_euler(&rotation_matrix, &order).unwrap();

        let expected_angles = vec![30.0_f64.to_radians(), 0.0, 0.0]; // Expected Euler angles

        for (res, exp) in result.iter().zip(expected_angles.iter()) {
            assert!((res - exp).abs() < 1e-5, "Expected {}, but got {}", exp, res);
        }
    }
}
