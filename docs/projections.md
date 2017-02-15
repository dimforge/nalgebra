# Projections
Projections in **nalgebra** are projections as commonly defined by the computer
graphics community. In particular, they are **not idempotent** as some
may be used to. Instead they are bijective mappings that transform a given
6-faced convex shape to the double unit cube centered at the origin (i.e. the
axis-aligned cube composed of points with coordinates ranging from $(-1, -1,
-1)$ to $(1, 1, 1)$). The resulting coordinates are usually called _Normalized
Device Coordinates_ (corresponding to the _clip-space_) by the computer
graphics community:

<center>
![projection](../img/projection.svg)
</center>

The actual shape to be transformed depends on the projection itself. Note that
projections implemented on **nalgebra** also flip the $\mathbf{z}$ axis. This
is a common convention in computer graphics applications for rendering with,
e.g., OpenGL, because the coordinate system of the screen is left-handed.


Currently, **nalgebra** defines only the 3D [orthographic
projection](#orthographic-projection) and the 3D [perspective
projection](#perspective-projection), aka., `Orthographic3` and `Perspective3`.
They both store a 4x4 homogeneous transformation matrix internally which can be
retrieved by-value using the `.unwrap()` or `.to_homogeneous()` methods. A
reference can be obtained with `.as_matrix()`. The projection matrix inverse
can be computed with the projection `.inverse()` method. Note that this will be
much more efficient than calling the inverse method on the raw homogeneous
`Matrix4`.

Projection types can transform points and vectors using the
`.project_point(...)` and `.project_vector(...)` methods. The latter ignores
the translational part of the projection because the input is a vector
(remember the [semantic difference](./points_and_transformations/#points)
between points and vectors). Because projections following our convention are
invertible, it is possible to apply the inverse projection to points using
`.unproject_point(...)`. This is typically used for screen-space coordinates to
view-space coordinates [conversion](/cg_recipes/#screen-space-to-view-space).

## Orthographic projection
An orthographic projection `Orthographic3` maps a rectangular axis-aligned
cuboid to the double unit cube centered at the origin. This is basically a
translation followed by a non-uniform scaling. An orthographic projection is
characterized by:

Property | Meaning
---------|--------
`left`   | The $\mathbf{x}$-coordinate of the cuboid leftmost face parallel to the $\mathbf{yz}$-plane.  |
`right`  | The $\mathbf{x}$-coordinate of the cuboid rightmost face parallel to the $\mathbf{yz}$-plane. |
`bottom` | The $\mathbf{y}$-coordinate of the cuboid leftmost face parallel to the $\mathbf{xz}$-plane.  |
`top`    | The $\mathbf{y}$-coordinate of the cuboid leftmost face parallel to the $\mathbf{xz}$-plane.   |
`znear`  | The distance between the viewer (the origin) and the closest face of the cuboid parallel to the $\mathbf{xy}$-plane. If used for a 3D rendering application, this is the closest clipping plane. |
`zfar`   | The distance between the viewer (the origin) and the furthest face of the cuboid parallel to the $\mathbf{xy}$-plane. If used for a 3D rendering application, this is the furthest clipping plane. |


The following example, shows the effect of an orthographic projections with its
`left`, `right`, `bottom`, `top`, `znear`, and `zfar` properties noted
respectively as $l$, $r$, $b$, $t$, $zn$, and $zf$:

<center>
![orthographic projection](../img/orthographic.svg)
</center>

```rust
// Arguments order: left, right, bottom, top, znear, zfar.
let proj = Orthographic3::new(1.0, 2.0, -3.0, -2.5, 10.0, 900.0);
let pt   = Point3::new(1.0, -3.0, 10.0);
let vec  = Vector3::new(21.0, 0.0, 0.0);

assert_eq!(orth.project_point(&pt),   Point3::new(-1.0, -1.0, -1.0));
assert_eq!(orth.project_vector(&vec), Vector3::new(42.0, 0.0, 0.0));
```


All properties can be read and modified. In-place modification is done with
methods starting with the `set_` name prefix, e.g., `.set_right(...)`. Instead
of recomputing the whole projection matrix, this will modify only the relevant
entries. Some setters combine two modifications at once for better efficiency:

Setter   | Meaning
---------|--------
`.set_left_and_right(...)` | Sets both left and right cuboid face coordinates simultaneously. |
`.set_bottom_and_top(...)` | Sets both bottom and top cuboid face coordinates simultaneously. |
`.set_znear_and_zfar(...)` | Sets both clipping planes simultaneously. |

## Perspective projection
A perspective projection `Perspective3` maps a frustrum to the double unit cube
centered at the origin. It is a non-linear transformation that uses homogeneous
coordinates to apply to each point a scale factor that depends on its distance
to the viewer.

<center>
![perspective projection](../img/perspective.svg)
</center>

The viewer of the perspective projection is always assumed to be located at the
origin and to look toward the $-z$ axis. Changing the viewer position and
orientation requires an additional isometry (in a separate data structure) to
form a [view-projection](cg_recipes/#build-a-mvp-matrix) transformation. A
perspective projection is characterized by:

Property | Meaning
---------|--------
`aspect` | The aspect ratio of the frustrum faces on the $\mathbf{xy}$-plane. This is division of the width by the height of any section (parallel to the $\mathbf{xy}$-plane) of the frustrum .
`fovy`   | The field of view along the $\mathbf{y}$ axis. This is the angle between uppermost and lowermost faces of the frustrum. |
`znear`  | The distance between the viewer (the origin) and the closest face of the frustrum parallel to the $\mathbf{xy}$-plane. If used for a 3D rendering application, this is the closest clipping plane. |
`zfar`   | The distance between the viewer (the origin) and the furthest face of the frustrum parallel to the $\mathbf{xy}$-plane. If used for a 3D rendering application, this is the furthest clipping plane. |

<center>
![perspective projection properties](../img/perspective_props.svg)
</center>

```rust
// Arguments order: aspect, fovy, znear, zfar.
let proj = Perspective3::new(16.0 / 9.0, 3.14 / 4.0, 1.0, 10000.0);
```

All properties can be read and modified. In-place modification is done with
methods starting with the `set_` name prefix, e.g., `.set_fovy(...)`. Instead
of recomputing the whole projection matrix, this will modify only the relevant
entries. The setter `.set_znear_and_zfar(...)` modify both clipping planes
simultaneously. This is more efficient than calling both `.set_znear(...)` and
`.set_zfar(...)` separately.
