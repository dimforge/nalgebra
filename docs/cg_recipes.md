# Recipes for Computer Graphics

This section describes how to perform some operations common for Computer
Graphics (CG). Note that while the 3D Computer Graphics community is used to
work almost exclusively with 4×4 matrices, **nalgebra** defines a wider number
of [transformation types](../points_and_transformations/#transformations) that
the user is strongly encouraged to use instead.


## Transformations using Matrix4
In the field of CG, a 4×4 matrix usually has a specific interpretation: it is
thought as a transformation matrix that mixes scaling (including shearing),
rotation and translation. Though using 4x4 matrices is convenient because most
3D transformations (including projections) can be represented using those
so-called *homogeneous coordinates*, they do not have provide strong guarantees
regarding their properties. For example, a method that takes a `Matrix4` in
argument cannot have the guarantee that it is a pure rotation, an
ispometry, or even an arbitrary but invertible transformation! That's why all
the [transformation types](../points_and_transformations/#transformations) are
recommended instead of raw matrices.

However, it is sometimes convenient to work directly with 4x4 matrices.
Especially for small applications where one wants to avoid the complexity of
having to select the right transformation type for the task at hand. Therefore,
**nalgebra** has a limited but useful support for transforming 3x3 matrices
(for 2D transformations) and 4x4 matrices (for 3d transformations).

#### Homogeneous raw transformation matrix creation
The following methods may be used on a `Matrix4`  to build a 4x4 homogeneous
raw transformation matrix.

Method                     | Description
---------------------------|--------------
`::new_scaling(s)`         | An uniform scaling matrix with scaling factor `s`. |
`::new_nonuniorm_scaling(vs)`  | A non-uniform scaling matrix with scaling factors along each coordinate given by the coordinates of the vector `vs`. |
`::new_translation(t)`     | A pure translation matrix specified by the displacement vector `t`. |
`::new_rotation_wrt_point(axang, pt)` | A composition of rotation and translation such that the point `pt` is left invariant. The rotational part is specified as a rotation axis multiplied by the rotation angle. |
`::from_scaled_axis(axang)`   | A pure rotation matrix specified by a rotation axis multiplied by the rotation angle. |
`::from_euler_angle(r, p, y)` | A pure rotation matrix from Euler angles applied in order: roll - pitch - yaw. |
`::new_orthographic(...)`     | An [orthographic](../projections/#orthographic-projection) projection matrix.  |
`::new_perspective(...)`      | A [perspective](../projections/#perspective-projection) projection matrix.     |
`::new_observer_frame(eye, target, up)` | A composition of rotation and translation corresponding to the local frame of a viewer standing at the point `eye` and looking toward `target`. The `up` direction is the vertical direction. |
`::look_at_rh(...)`           | A right-handed look-at view matrix. |
`::look_at_lh(...)`           | A left-handed look-at view matrix.  |

Note that a few of those functions are also defined for `Matrix3` which can
hold the homogeneous coordinates of 2D transformations.

#### Homogeneous raw transformation matrix modification
Once created, a `Matrix4` (or `Matrix3` for 2D transformations) can be modified
by appending or prepending transformations. The function signature follow the
same pattern as the transformation matrix creation functions listed above.
In-place appending and prepending are supported and have a name with a `_mut`
suffix, e.g., `.append_scaling_mut(...)` instead of `.append_scaling(...)`.

Method                            | Description
----------------------------------|--------------
`.append_scaling(s)`              | Appends to `self` a uniform scaling with scaling factor `s`.
`.prepend_scaling(s)`             | Prepends to `self` a uniform scaling with scaling factor `s`.
`.append_nonuniform_scaling(vs)`  | Appends to `self` a non-uniform scaling with scaling factors along each coordinate given by the coordinates of `vs`. |
`.prepend_nonuniform_scaling(vs)` | Prepends to `self` a non-uniform scaling with scaling factors along each coordinate given by the coordinates of `vs`. |
`.append_translation(t)`          | Appends to `self` a translation specified by the vector `t`.
`.prepend_translation(t)`         | Prepends to `self` a translation specified by the vector `t`.

Note that there isn't any method to append or prepend a rotation. That is because a
specific method does not provide any performance benefit. Instead, you may
explicitly construct an homogeneous rotation matrix using, e.g.,
`::from_scaled_axis`, and then multiply the result with the matrix you want it
appended or prepended to.

<ul class="nav nav-tabs">
  <li class="active"><a id="tab_nav_link" data-toggle="tab" href="#transform_matrix4">Example</a></li>

  <div class="btn-primary" onclick="window.open('https://raw.githubusercontent.com/sebcrozet/nalgebra/master/examples/transform_matrix4.rs')"></div>
</ul>

<div class="tab-content" markdown="1">
  <div id="transform_matrix4" class="tab-pane in active">
```rust
// Create a uniform scaling matrix with scaling factor 2.
let mut m =  Matrix4::new_scaling(2.0);

assert_eq!(m.transform_vector(&Vector3::x()), Vector3::x() * 2.0);
assert_eq!(m.transform_vector(&Vector3::y()), Vector3::y() * 2.0);
assert_eq!(m.transform_vector(&Vector3::z()), Vector3::z() * 2.0);

// Append a nonuniform scaling in-place.
m.append_nonuniform_scaling_mut(&Vector3::new(1.0, 2.0, 3.0));

assert_eq!(m.transform_vector(&Vector3::x()), Vector3::x() * 2.0);
assert_eq!(m.transform_vector(&Vector3::y()), Vector3::y() * 4.0);
assert_eq!(m.transform_vector(&Vector3::z()), Vector3::z() * 6.0);

// Append a translation out-of-place.
let m2 = m.append_translation(&Vector3::new(42.0, 0.0, 0.0));

assert_eq!(m2.transform_point(&Point3::new(1.0, 1.0, 1.0)), Point3::new(42.0 + 2.0, 4.0, 6.0));

// Create rotation.
let rot        = Matrix4::from_scaled_axis(&Vector3::x() * 3.14);
let rot_then_m = m * rot; // Right-multiplication is equivalent to prepending `rot` to `m`.
let m_then_rot = rot * m; // Left-multiplication is equivalent to appending `rot` to `m`.

let pt = Point3::new(1.0, 2.0, 3.0);

assert_relative_eq!(m.transform_point(&rot.transform_point(&pt)), rot_then_m.transform_point(&pt));
assert_relative_eq!(rot.transform_point(&m.transform_point(&pt)), m_then_rot.transform_point(&pt));
```
  </div>
</div>

#### Using raw transformation matrices on points and vectors
Homogeneous raw transformation matrix do not have a compatible dimensions for
multiplication with a vector of a point. For example a 3D transformation
represented by a `Matrix4` cannot multiply a 3D vector represented by a
`Vector3` (because the matrix has 4 columns while the vector has only 3 rows).
There are two main ways to deal with this issue:

1. Use the `.transform_vector(...)` and `.transform_point(...)` methods that
   directly take a `Vector3` and `Point3` as argument.
2. Use homogeneous coordinates for vectors and points as well. In that case, a
   3D vector `Vector3::new(x, y, z)` should be given given a fourth coordinate
   set to zero, i.e. `Vector4::new(x, y, z, 0.0)` while a 3D point
   `Point3::new(x, y, z)` should be represented as a vector with its fourth
   coordinate equal to one, i.e., `Vector4::new(x, y, z, 1.0)`. Then the
   `Matrix4` can multiply the augmented vector directly.


<ul class="nav nav-tabs">
  <li class="active"><a id="tab_nav_link" data-toggle="tab" href="#transform_vector_point3">Example</a></li>

  <div class="btn-primary" onclick="window.open('https://raw.githubusercontent.com/sebcrozet/nalgebra/master/examples/transform_vector_point3.rs')"></div>
</ul>

<div class="tab-content" markdown="1">
  <div id="transform_vector_point3" class="tab-pane in active">
```rust
let mut m = Matrix4::new_rotation_wrt_point(Vector3::x() * 1.57, Point3::new(1.0, 2.0, 1.0));
m.append_scaling_mut(2.0);

let point1             = Point3::new(2.0, 3.0, 4.0);
let homogeneous_point2 = Vector4::new(2.0, 3.0, 4.0, 1.0);

// First option: use the dedicated `.transform_point(...)` method.
let transformed_point1 = m.transform_point(&point1);
// Second option: use the homogeneous coordinates of the point.
let transformed_homogeneous_point2 = m * homogeneous_point2;

// Recover the 3D point from its 4D homogeneous coordinates.
let transformed_point2 = Point3::from_homogeneous(transformed_homogeneous_point2);

// Check that transforming the 3D point with the `.transform_point` method is
// indeed equivalent to multiplying its 4D homogeneous coordinates by the 4x4
// matrix.
assert_eq!(transformed_point1, transformed_point2.unwrap());
```
  </div>
</div>


## Build a MVP matrix
The Model-View-Projection matrix is the common denomination of the composition
of three transformations:

1. The <u>model transformation</u> gives its orientation and position to an object
   in the 3D scene. It is different for every object.
2. The <u>view transformation</u> that moves any point of the scene into the local
   coordinate of the camera. It is the same for every object on the scene, but
   different for every camera.
3. The <u>projection</u> that translates and stretches the displayable part of
   the scene so that it fits into the double unit cube (aka. Normalized Device
   Coordinates). We already discussed it [there](../projections). There is
   usually only one projection per display.

Note that it is also common to construct only a View-Projection matrix and let
the graphics card combine it with the model transformation in shaders. For
completeness, our example will deal with the model transformation as well.


The model and view transformations are direct isometries. Thus, we can simply
use the dedicated `Isometry3` type. The projection is not an isometry and
requires the use of a raw `Matrix4` or a dedicated
[projection](../projections) type like `Perspective3`.

<ul class="nav nav-tabs">
  <li class="active"><a id="tab_nav_link" data-toggle="tab" href="#mvp">Example</a></li>

  <div class="btn-primary" onclick="window.open('https://raw.githubusercontent.com/sebcrozet/nalgebra/master/examples/mvp.rs')"></div>
</ul>

<div class="tab-content" markdown="1">
  <div id="cargo" class="tab-pane in active">
```rust
// Our object is translated along the x axis.
let model = Isometry3::new(Vector3::x(), na::zero());

// Our camera looks toward the point (1.0, 0.0, 0.0).
// It is located at (0.0, 0.0, 1.0).
let eye    = Point3::new(0.0, 0.0, 1.0);
let target = Point3::new(1.0, 0.0, 0.0);
let view   = Isometry3::look_at_rh(&eye, &target, &Vector3::y());

// A perspective projection.
let projection = Perspective3::new(16.0 / 9.0, 3.14 / 2.0, 1.0, 1000.0);

// The combination of the model with the view is still an isometry.
let model_view = view * model;

// Convert everything to a `Matrix4` so that they can be combined.
let mat_model_view = model_view.to_homogeneous();

// Combine everything.
let model_view_projection = projection.as_matrix() * mat_model_view;
```
  </div>
</div>

Of course, the last four `let` will usually be written in a single line:

```rust
let eye    = Point3::new(0.0, 0.0, 1.0);
let target = Point3::new(1.0, 0.0, 0.0);
let view   = Isometry3::look_at_rh(&eye, &target, &Vector3::y());

let model      = Isometry3::new(Vector3::x(), na::zero());
let projection = Perspective3::new(16.0 / 9.0, 3.14 / 2.0, 1.0, 1000.0);
let model_view_projection = projection.unwrap() * (view * model).to_homogeneous();
```

## Screen-space to view-space
It is the projection matrix task to stretch and translate the displayable 3D
objects into the [double unit cube](projections), i.e., it transforms points
from view-coordinates (the camera local coordinate system) to Normalized Device
Coordinates (aka. clip-space).  Then, the screen itself will contain everything
that can be seen from the cube's face located on the plane $z = -1$.
Therefore, a whole line (in clip-space) parallel to the $\mathbf{z}$ axis will
be mapped to a single point on screen-space (the display device's 2D coordinate
system). The following shows one such line $\mathcal{L}$ in view space,
normalized device coordinates, and screen-space.  More details about the
different coordinate systems use in compute graphics can be found
[there](http://learnopengl.com/#!Getting-started/Coordinate-Systems).

<center>
![3D-space to screen-space](../img/view_to_screen_space.svg)
</center>

Now observe that there is a bijective relationship between each point in
screen-space and each line parallel to the $\mathbf{z}$ axis in clip-space.
Moreover, both the perspective and orthographic projections are bijective and
map lines to lines.  It is thus possible to perform a so-called _unprojection_,
i.e., from a 2D point in screen-space compute the corresponding 3D line in
view-space. Typically, this line can then be used for picking using [ray
casting](https://ncollide.org/geometric_queries/#ray-casting).  The next
example takes a point on a screen of size $800 \times 600$ and retrieves the
corresponding line in view-space. It follows three steps:

1. Convert the point from screen-space to two points in clip-space. One will
   lie on the near-plane with $z = -1$ and the other on the far-plane with $z = 1$.
2. Apply the inverse projection both points.
3. Compute the parameters of the line that passes through those two points.

<ul class="nav nav-tabs">
  <li class="active"><a id="tab_nav_link" data-toggle="tab" href="#screen_to_view_coords">Example</a></li>

  <div class="btn-primary" onclick="window.open('https://raw.githubusercontent.com/sebcrozet/nalgebra/master/examples/screen_to_view_coords.rs')"></div>
</ul>

<div class="tab-content" markdown="1">
  <div id="screen_to_view_coords" class="tab-pane in active">
```rust
let projection   = Perspective3::new(800.0 / 600.0, 3.14 / 2.0, 1.0, 1000.0);
let screen_point = Point2::new(10.0f32, 20.0);

// Compute two points in clip-space.
// "ndc" = normalized device coordinates.
let near_ndc_point = Point3::new(screen_point.x / 800.0, screen_point.y / 600.0, -1.0);
let far_ndc_point  = Point3::new(screen_point.x / 800.0, screen_point.y / 600.0,  1.0);

// Unproject them to view-space.
let near_view_point = projection.unproject_point(&near_ndc_point);
let far_view_point  = projection.unproject_point(&far_ndc_point);

// Compute the view-space line parameters.
let line_location  = near_view_point;
let line_direction = Unit::new_normalize(far_view_point - near_view_point);
```
  </div>
</div>

The resulting 3D line will be in the local space of the camera. Thus, it might
be useful to multiply its location an direction by the inverse view matrix in
order to obtain coordinates in world-space. The same procedure will work with
any other straight line preserving projection as well, e.g., with the orthographic
projection.

## Conversions for shaders
Shaders don't understand the high-level types defined by **nalgebra**.
Therefore, you will usually need to convert your data into pointers to a
contiguous array of floating point numbers with components arranged in a
specific way in-memory. Using the `.as_slice()` method of raw matrices and
vectors, one can retrieve a reference to a contiguous array containing all
components in column-major order. Note that this method will not exist for
matrices that do not own their data, e.g. [matrix
slices](../vectors_and_matrices/#matrix-slicing).

<ul class="nav nav-tabs">
  <li class="active"><a id="tab_nav_link" data-toggle="tab" href="#raw_pointer">Example</a></li>

  <div class="btn-primary" onclick="window.open('https://raw.githubusercontent.com/sebcrozet/nalgebra/master/examples/raw_pointer.rs')"></div>
</ul>

<div class="tab-content" markdown="1">
  <div id="raw_pointer" class="tab-pane in active">
```rust
let v = Vector3::new(1.0f32, 0.0, 1.0);
let p = Point3::new(1.0f32, 0.0, 1.0);
let m = na::one::<Matrix3<f32>>();

// Convert to arrays.
let v_array = v.as_slice();
let p_array = p.coords.as_slice();
let m_array = m.as_slice();

// Get data pointers.
let v_pointer = v_array.as_ptr();
let p_pointer = p_array.as_ptr();
let m_pointer = m_array.as_ptr();

/* Then pass the raw pointers to some graphics API. */
```
  </div>
</div>

Higher-level types of transformations like `Rotation3`, `Similarity3` cannot
be converted into arrays directly so you will have to convert them to raw
matrices first. The underlying 3×3 or 4×4 matrix of `RotationMatrix3` and
`RotationMatrix2` can be directly retrieved by the `.matrix()` method. All the
other transformations must first be converted into their [homogeneous
coordinates](../points_and_transformations/#homogeneous-coordinates) representation
using the method `.to_homogeneous()`. This will return a `Matrix2` or a
`Matrix3` that may then be reinterpreted as arrays.

<ul class="nav nav-tabs">
  <li class="active"><a id="tab_nav_link" data-toggle="tab" href="#transformation_pointer">Example</a></li>

  <div class="btn-primary" onclick="window.open('https://raw.githubusercontent.com/sebcrozet/nalgebra/master/examples/transformation_pointer.rs')"></div>
</ul>

<div class="tab-content" markdown="1">
  <div id="transformation_pointer" class="tab-pane in active">
```rust
let iso = Isometry3::new(na::zero(), Vector3::new(1.0f32, 0.0, 1.0));

// Compute the homogeneous coordinates first.
let iso_matrix  = iso.to_homogeneous();
let iso_array   = iso_matrix.as_slice();
let iso_pointer = iso_array.as_ptr();

/* Then pass the raw pointer to some graphics API. */
```
  </div>
</div>
