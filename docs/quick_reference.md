# Quick reference

Free functions are noted with a leading `::` while methods start with a dot.
Most types are type aliases. Refer to the [API
documentation](../rustdoc_nalgebra) for details about the functions arguments
and type parameters.

* [Matrices and vectors](#matrices-and-vectors)
    * [Construction](#construction), [Common methods](#common-methods), [Slicing](#slicing)
    * [Decompositions](#decompositions)
    * [Computer graphics](#computer-graphics)
* [Geometry](#geometry)
    * [Points and transformations](#geometry)
    * [Projections](#projections)
* [Data storages and allocators](#data-storage-and-allocators)

Serialization with [serde](https://serde.rs) can be enabled by enabling the
**serde-serialize** feature.

### Matrices and vectors

* Within this reference, `N` is the scalar type, `R`, `C` and `D` are type-level integers.  
* Matrices are stored in column-major order.
* Vectors are type aliases for matrices with only one column or one row.
* Overloaded operators: `*`, `/`, `+`, `-` (binary and unary), and corresponding assignement operators.
* Comparison operators `==`, `>=`, `<=`, `>`, `<`,  using column-major lexicographic ordering.
* Mutable and non-mutable indexing: `the_vector[usize]` and `the_matrix[(usize, usize)]`.
* All angles are in radian.

-----

`Matrix<...>`                                       <span style="float:right;">Generic matrix type.</span><br/>
`Matrix1<N> .. Matrix6<N>`, `MatrixN<N, D>`         <span style="float:right;">Statically-sized square matrix.</span><br/>
`Matrix1x2<N> .. Matrix6x5<N>`, `MatrixNM<N, R, C>` <span style="float:right;">Statically-sized rectangular matrix.</span><br/>
`DMatrix<N>`                                        <span style="float:right;">Dynamically-sized matrix.</span>

<br/>

`Vector1<N> .. Vector6<N>`, `VectorN<N, D>` <span style="float:right;">Statically-sized column vector.</span><br/>
`DVector<N>`                                <span style="float:right;">Dynamically-sized column vector.</span>

<br/>

`RowVector1<N> .. RowVector6<N>`, `RowVectorN<N, D>` <span style="float:right;">Statically-sized row vector.</span><br/>
`RowDVector<N>`                                      <span style="float:right;">Dynamically-sized row vector.</span>

<br/>

`Unit<T>`                                            <span style="float:right;">Wrapper that ensures the underlying value of type `T` is normalized.</span>


-----

#### Construction
The exact number of argument, and the existence of some constructors depend on
the result matrix type. For example, dynamically-sized or 2D vectors do not
have a `::z()` constructors. In the following "Low-dimensional" means with
dimensions 1 to 6 for vectors and 1x1 to 6x6 for matrices.


`::x(), ::y(), ::z(), ::w(), ::a(), ::b()` <span style="float:right;">Low-dimensional vector with one component set to 1 and others to 0.</span><br/>
`::x_axis() .. ::b_axis()`                 <span style="float:right;">Low-dimensional vector wrapped in `Unit<...>`.</span><br/>
`::new(x, y, .. b)`                        <span style="float:right;">Low-dimensional vector constructed from its components values.</span><br/>
`::new(m11, m12, .. m66)`                  <span style="float:right;">Low-dimensional matrix constructed from its components values.</span><br/>

<br/>

`::new_uninitialized(...)`     <span style="float:right;">_[unsafe]_ Matrix with uninitialized components.</span><br/>
`::new_random(...)`            <span style="float:right;">Matrix filled with random values.</span><br/>
`::identity(...)`              <span style="float:right;">The identity matrix.</span><br/>
`::from_element(...)`          <span style="float:right;">Matrix filled with the given element.</span><br/>
`::from_iterator(...)`         <span style="float:right;">Matrix filled with the content of the given iterator.</span><br/>
`::from_row_slice(...)`        <span style="float:right;">Matrix filled with the content of the components given in **row-major** order.</span><br/>
`::from_column_slice(...)`     <span style="float:right;">Matrix filled with the content of the components given in **column-major** order.</span><br/>
`::from_fn(...)`               <span style="float:right;">Matrix filled with the result of a closure called for each entry.</span><br/>
`::from_diagonal(...)`         <span style="float:right;">Diagonal matrix with the given diagonal vector.</span><br/>
`::from_diagonal_element(...)` <span style="float:right;">Diagonal matrix with the diagonal filled with one value.</span><br/>
`::from_rows(...)`             <span style="float:right;">Matrix formed by the concatenation of the given rows.</span><br/>
`::from_columns(...)`          <span style="float:right;">Matrix formed by the concatenation of the given columns.</span><br/>

<br/>

`Zero::zero()`         <span style="float:right;">Matrix filled with zeroes.</span><br/>
`One::one()`           <span style="float:right;">The identity matrix.</span><br/>
`Bounded::min_value()` <span style="float:right;">Matrix filled with the min value of the scalar type.</span><br/>
`Bounded::max_value()` <span style="float:right;">Matrix filled with the max value of the scalar type.</span><br/>
`Rand::rand(...)`      <span style="float:right;">Matrix filled with random values.</span><br/>

----

#### Common methods

`.len()`     <span style="float:right;">The number of components.</span><br/>
`.shape()`   <span style="float:right;">The number of rows and columns.</span><br/>
`.nrows()`   <span style="float:right;">The number of rows.</span><br/>
`.ncols()`   <span style="float:right;">The number of columns.</span><br/>
`.strides()` <span style="float:right;">Number of skipped original rows/columns in-between each row/column of a slice.</span><br/>

<br/>

`.iter()`                  <span style="float:right;">An iterator through the matrix components in column-major order.</span><br/>
`.iter_mut()`              <span style="float:right;">A mutable iterator through the matrix components in column-major order.</span><br/>
`.get_unchecked(i, j)`     <span style="float:right;">_[unsafe]_ The matrix component at row `i` and column `j`. No bound-checking.</span><br/>
`.get_unchecked_mut(i, j)` <span style="float:right;">_[unsafe]_ The mutable matrix component at row `i` and column `j`. No bound-checking.</span><br/>
`.swap_unchecked(i, j)`    <span style="float:right;">_[unsafe]_ Swaps two components. No bound-checking.</span><br/>
`.as_slice()`              <span style="float:right;">Reference to the internal column-major array of component.</span><br/>
`.as_mut_slice()`          <span style="float:right;">Mutable reference to the internal column-major array of component.</span><br/>

<br/>

`.copy_from(m)`   <span style="float:right;">Copies the content of another matrix with the same shape.</span><br/>
`.fill(e)`        <span style="float:right;">Sets all components to `e`.</span><br/>
`.map(f)`         <span style="float:right;">Applies `f` to each component and stores the results on a new matrix.</span><br/>
`.zip_map(m2, f)` <span style="float:right;">Applies `f` to the pair of component from `self` and `m2` and stores the results on a new matrix.</span><br/>

<br/>

`.relative_eq(...)`       <span style="float:right;">Componentwise approximate matrix equality.</span><br />
`.component_mul(rhs)`     <span style="float:right;">Componentwise multiplication (aka. Hadamard product).</span><br />
`.component_mul_mut(rhs)` <span style="float:right;">In-place componentwise multiplication (aka. Hadamard product).</span><br />
`.component_div(rhs)`     <span style="float:right;">Componentwise division.</span><br />
`.component_div_mut(rhs)` <span style="float:right;">In-place componentwise division.</span><br />

<br/>

`.transpose()`       <span style="float:right;">Matrix transposition.</span><br />
`.transpose_mut()`   <span style="float:right;">In-place matrix transposition.</span><br />
`.try_inverse()`     <span style="float:right;">Matrix inverse. Returns `None` if it fails.</span><br />
`.try_inverse_mut()` <span style="float:right;">In-place matrix inverse. Returns `false` if it fails.</span><br />

<br/>

`.diagonal()`          <span style="float:right;">The matrix diagonal.</span><br />
`.abs()`               <span style="float:right;">The absolute value of this matrix components.</span><br />
`.determinant()`       <span style="float:right;">The matrix determinant.</span><br />
`.trace()`             <span style="float:right;">The trace of this matrix.</span><br />
`.norm()`              <span style="float:right;">The L2 norm.</span><br />
`.norm_squared()`      <span style="float:right;">The squared L2 norm.</span><br />
`.normalize()`         <span style="float:right;">Normalization using the L2 norm.</span><br />
`.normalize_mut()`     <span style="float:right;">In-place normalization using the L2 norm.</span><br />
`.try_normalize()`     <span style="float:right;">Normalization. Returns `None` if the norm is too small.</span><br />
`.try_normalize_mut()` <span style="float:right;">In-place normalization.  Returns `None` if the norm is too small.</span><br />

<br/>

`.dot(rhs)`    <span style="float:right;">Vector dot product.</span><br />
`.tr_dot(rhs)` <span style="float:right;">Vector dot product between `self.transpose()` and `rhs`.</span><br />
`.perp(rhs)`   <span style="float:right;">2D cross product, i.e., determinant of the matrix formed by two 2D column vectors.</span><br />
`.cross(rhs)`  <span style="float:right;">3D cross product.</span><br />
`.angle(rhs)`  <span style="float:right;">Smallest angle between two vectors.</span><br />

<br/>

`.is_square()`             <span style="float:right;">Tests if `self` is square.</span><br />
`.is_identity()`           <span style="float:right;">Tests if `self` is the identity matrix.</span><br />
`.is_orthogonal()`         <span style="float:right;">Tests if `self.transpose() * self` is the identity.</span><br />
`.is_special_orthogonal()` <span style="float:right;">Tests if `self.is_orthogonal()` and has a determinant equal to 1.</span><br />
`.is_invertible()`         <span style="float:right;">Tests if `self` is invertible.</span><br />

---------

#### Slicing
Slice are references to sub-matrices. They do not own their data and cannot be
converted to arrays as their data buffer may not be contiguous in memory.


`.row(...)`                       <span style="float:right;">A matrix row.</span><br />
`.rows(...)`                      <span style="float:right;">Several consecutive rows.</span><br />
`.rows_with_step(...)`            <span style="float:right;">Several non-consecutive rows.</span><br />
`.fixed_rows::<D>(...)`           <span style="float:right;">A compile-time number of consecutive rows.</span><br />
`.fixed_rows_with_step::<D>(...)` <span style="float:right;">A compile-time number of non-consecutive rows.</span><br />

<br/>

`.column(...)`                       <span style="float:right;">A matrix column.</span><br />
`.columns(...)`                      <span style="float:right;">Several consecutive columns.</span><br />
`.columns_with_step(...)`            <span style="float:right;">Several non-consecutive columns.</span><br />
`.fixed_columns::<D>(...)`           <span style="float:right;">A compile-time number of consecutive columns.</span><br />
`.fixed_columns_with_step::<D>(...)` <span style="float:right;">A compile-time number of non-consecutive columns.</span><br />

<br/>

`.slice(...)`                          <span style="float:right;">Consecutive rows and columns.</span><br />
`.slice_with_steps(...)`               <span style="float:right;">Non consecutive rows and columns.</span><br />
`.fixed_slice::<R, C>(...)`            <span style="float:right;">Compile-time number of consecutive rows and columns.</span><br />
`.fixed_slice_with_steps::<R, C>(...)` <span style="float:right;">Compile-time number of non consecutive rows and columns.</span><br />

-----

#### Decompositions

`.qr()`                 <span style="float:right;">QR decomposition.</span><br />
`.eig(...)`             <span style="float:right;">Eigenvalue and eigenvectors computation.</span><br />
`.cholesky()`           <span style="float:right;">Cholesky factorization.</span><br />
`.cholesky_unchecked()` <span style="float:right;">Cholesky factorization without checking the matrix symmetry.</span><br />
`.hessenberg()`         <span style="float:right;">Hessenberg form computation.</span><br />

------

#### Computer graphics
Those are constructors and methods useful for the computer graphics community
that work with homogeneous matrix coordinates, e.g., 4x4 matrices for 3D
transformations. Homogeneous matrix coordinates are expected to be multiplied
by homogeneous vectors (the vector goes on the right-hand-side of the
operator).

`::new_scaling(...)`            <span style="float:right;">An uniform scaling matrix.</span><br />
`::new_nonuniform_scaling(...)` <span style="float:right;">A nonuniform scaling matrix.</span><br />
`::new_translation(...)`        <span style="float:right;">A translation matrix.</span><br />
`::new_rotation(...)`           <span style="float:right;">A rotation matrix from an axis multiplied by an angle.</span><br />
`::new_rotation_wrt_point(...)` <span style="float:right;">An isometry matrix that lets the given point invariant.</span><br />
`::from_scaled_axis(...)`       <span style="float:right;">A rotation matrix from an axis multiplied by an angle.</span><br />
`::from_euler_angles(...)`      <span style="float:right;">A rotation matrix from euler angles (roll → pitch → yaw).</span><br />
`::from_axis_angle(...)`        <span style="float:right;">A rotation matrix from an axis and an angle.</span><br />
`::new_orthographic(...)`       <span style="float:right;">An orthographic projection matrix.</span><br />
`::new_perspective(...)`        <span style="float:right;">A perspective projection matrix.</span><br />
`::new_observer_frame(...)`     <span style="float:right;">The local coordinate system of a player looking toward a given point.</span><br />
`::look_at_rh(...)`             <span style="float:right;">A right-handed look-at matrix.</span><br />
`::look_at_lh(...)`             <span style="float:right;">A left-handed look-at matrix.</span><br />

<br/>

`.append_scaling(...)`                 <span style="float:right;">Applies an uniform scaling after `self`.</span><br />
`.append_scaling_mut(...)`             <span style="float:right;">Applies in-place an uniform scaling after `self`.</span><br />
`.prepend_scaling(...)`                <span style="float:right;">Applies an uniform scaling before`self`.</span><br />
`.prepend_scaling_mut(...)`            <span style="float:right;">Applies in-place an uniform scaling before `self`.</span><br />
`.append_nonuniform_scaling(...)`      <span style="float:right;">Applies a non-uniform scaling after `self`.</span><br />
`.append_nonuniform_scaling_mut(...)`  <span style="float:right;">Applies in-place a non-uniform scaling after `self`.</span><br />
`.prepend_nonuniform_scaling(...)`     <span style="float:right;">Applies a non-uniform scaling before`self`.</span><br />
`.prepend_nonuniform_scaling_mut(...)` <span style="float:right;">Applies in-place a non-uniform scaling before `self`.</span><br />
`.append_translation(...)`             <span style="float:right;">Applies a translation after `self`.</span><br />
`.append_translation_mut(...)`         <span style="float:right;">Applies in-place a translation after `self`.</span><br />
`.prepend_translation(...)`            <span style="float:right;">Applies a translation before `self`.</span><br />
`.prepend_translation_mut(...)`        <span style="float:right;">Applies in-place a translation before `self`.</span><br />

-----

### Geometry

* Most geometric entities are just wrappers for a matrix (or a vector).
* Points have a very different semantic than vectors.
* Transformation types are preferred over raw matrices as they allow
  optimization because of their intrinsic properties.
* Use `.to_homogeneous()` pour obtain the corresponding homogeneous raw matrix
  of any geometric entity.
* Transformations of different types can be multiplied together. The result
  type is automatically inferred.
* Transforming a point or vector can be done by multiplication. The
  transformation goes on the left-hand-side.
* In the following, notations like `Transform2/3<N>` means "either
  `Transform2<N>` or `Transform3<N>`".

`Point2/3<N>`       <span style="float:right;">A location in space.</span><br/>
`Quaternion<N>`     <span style="float:right;">A general quaternion.</span><br/>
`Rotation2/3<N>`    <span style="float:right;">A rotation matrix.</span><br/>
`UnitComplex<N>`    <span style="float:right;">A 2D rotation represented as a unit complex number.</span><br/>
`UnitQuaternion<N>` <span style="float:right;">A 3D rotation represented as a unit quaternion.</span><br/>
`Translation2/3<N>` <span style="float:right;">A translation (stored as a vector).</span><br/>
`Isometry2/3<N>`    <span style="float:right;">A 2D/3D isometry containing an unit complex/quaternion.</span><br/>
`Similarity2/3<N>`  <span style="float:right;">A 2D/3D similarity containing an unit complex/quaternion.</span><br/>
`Affine2/3<N>`      <span style="float:right;">An affine transformation stored as an homogeneous matrix.</span><br/>
`Projective2/3<N>`  <span style="float:right;">An invertible transformation stored as an homogeneous matrix.</span><br/>
`Transform2/3<N>`   <span style="float:right;">A general transformation stored as an homogeneous matrix.</span><br/>

-----

#### Projections

Projections follow the behavior expected by Computer Graphics community, i.e.,
they are invertible transformation from a convex shape to a unit cube centered
at the origin.

`Perspective3<...>`    <span style="float:right;">3D perspective projection matrix using homogeneous coordinates.</span><br/>
`Orthographic3<...>`    <span style="float:right;">3D orthographic projection matrix using homogeneous coordinates.</span><br/>

-----

#### Base types
Base types are generic wrt. the dimension and/or the data storage type. They
should not be used directly, prefer type aliases shown in the previous section
instead.

`PointBase<...>`                 <span style="float:right;">A location in space.</span><br/>
`QuaternionBase<...>`            <span style="float:right;">A general quaternion.</span><br/>
`RotationBase<...>`              <span style="float:right;">A rotation matrix.</span><br/>
`TranslationBase<...>`           <span style="float:right;">A translation vector.</span><br/>
`IsometryBase<...>`              <span style="float:right;">An isometry containing an abstract rotation type.</span><br/>
`SimilarityBase<...>`            <span style="float:right;">A similarity containing an abstract rotation type.</span><br/>
`TransformBase<..., Affine>`     <span style="float:right;">An affine transformation stored as an homogeneous matrix.</span><br/>
`TransformBase<..., Projective>` <span style="float:right;">An invertible transformation stored as an homogeneous matrix.</span><br/>
`TransformBase<..., General>`    <span style="float:right;">A general transformation stored as an homogeneous matrix.</span><br/>
`PerspectiveBase<...>`           <span style="float:right;">A perspective projection matrix.</span><br/>
`OrthographicBase<...>`          <span style="float:right;">An orthographic projection matrix.</span><br/>

-----

### Data storage and allocators

* Data storages provide access to a matrix shape and its components.
* The last type parameter of the generic `Matrix<...>` type is the data storage.
* Allocators provide a way to allocate a data storage type that depends
  on whether the matrix shape is statically known.
* Transformation types require a data storage that implements `OwnedStorage<...>`.


#### Traits
`Storage<...>`        <span style="float:right;">Structures that give access to a buffer of data.</span><br/>
`StorageMut<...>`     <span style="float:right;">Structures that give access to a mutable buffer of data.</span><br/>
`OwnedStorage<...>`   <span style="float:right;">Structures that uniquely own their mutable and contiguous data buffer.</span><br/>
`Allocator<...>`      <span style="float:right;">Structures that can allocate a data buffer.</span><br/>
`OwnedAllocator<...>` <span style="float:right;">An allocator for a specific owned data storage.</span><br/>

-----

#### Implementors
`MatrixArray<...>`      <span style="float:right;">A stack-allocated owned data storage.</span><br/>
`MatrixVec<...>`        <span style="float:right;">A heap-allocated owned data storage.</span><br/>
`MatrixSlice<...>`      <span style="float:right;">A non-mutable reference to a piece of another data storage.</span><br/>
`MatrixSliceMut<...>`   <span style="float:right;">A mutable reference to a piece of another data storage.</span><br/>
`DefaultAllocator<...>` <span style="float:right;">Allocates a `MatrixArray` for compile-time-sized matrices, and a `MatrixVec` otherwise.</span><br/>
