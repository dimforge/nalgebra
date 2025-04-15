# How to use nalgebra's Euler Angles in your code

## Rotation on OX axis

```rs
let rotation_matrix = rotation_x(&45.0);
println!("{:?}", rotation_matrix);
```


## Rotation on OY axis

```rs
let rotation_matrix = rotation_y(&45.0);
println!("{:?}", rotation_matrix);
```


## Rotation on OZ axis

```rs
let rotation_matrix = rotation_z(&45.0);
println!("{:?}", rotation_matrix);
```

## Euler to rotation matrix

```rs
let angles = [45.0, 30.0, 60.0];
let order = ['X', 'Y', 'Z'];
match euler_to_rotation_matrix(&angles, &order) {
    Ok(matrix) => println!("{:?}", matrix),
    Err(err) => println!("Error: {}", err),
}
```

## Rotation matrix to euler


```rs
let rotation_matrix = vec![
    vec![0.5, -0.5, 0.7071],
    vec![0.7071, 0.7071, 0.0],
    vec![-0.5, -0.5, 0.7071]
];
let order = ['X', 'Y', 'Z'];
match rotation_matrix_to_euler(&rotation_matrix, &order) {
    Ok(angles) => println!("{:?}", angles),
    Err(err) => println!("Error: {}", err),
}
```