use crate::base::{Const, Scalar, ToTypenum};
use crate::geometry::{Point, Point2, Point3};
use typenum::{self, Cmp, Greater};

macro_rules! impl_swizzle {
    ($( where $BaseDim: ident: $( $name: ident() -> $Result: ident[$($i: expr_2021),+] ),+ ;)* ) => {
        $(
            $(impl_swizzle!(@method $name, $Result, $($i),+);)*
        )*
    };

    // Two-component swizzles
    (@method xx, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point by duplicating the x-component.
        ///
        /// This is a **swizzle** operation, a technique commonly used in graphics programming
        /// to rearrange or duplicate vector/point components. The name `xx` indicates that
        /// both output components come from the x-axis (index 0).
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.xx()` creates both components from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL, where
        /// swizzling is a fundamental operation for vector manipulation.
        ///
        /// # Examples
        ///
        /// Basic usage - duplicating the x-component:
        ///
        /// ```
        /// use nalgebra::Point2;
        ///
        /// let point = Point2::new(5.0, 3.0);
        /// let swizzled = point.xx();
        ///
        /// assert_eq!(swizzled, Point2::new(5.0, 5.0));
        /// ```
        ///
        /// Practical use case - creating uniform scaling from a single dimension:
        ///
        /// ```
        /// use nalgebra::Point2;
        ///
        /// let scale_factor = Point2::new(2.0, 999.0);
        /// // Extract just the x-component as a uniform 2D scale
        /// let uniform_scale = scale_factor.xx();
        ///
        /// assert_eq!(uniform_scale, Point2::new(2.0, 2.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xy()`](Self::xy) - preserves original x and y components
        /// - [`yx()`](Self::yx) - swaps x and y components
        /// - [`yy()`](Self::yy) - duplicates the y-component
        /// - [`xxx()`](Self::xxx) - creates a 3D point from duplicated x
        #[inline]
        #[must_use]
        pub fn xx(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U0, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    (@method xy, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point from the x and y components.
        ///
        /// This is a **swizzle** operation that extracts the first two components of a point.
        /// For 2D points, this returns an identical copy. For higher-dimensional points,
        /// this extracts just the x and y coordinates.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.xx()` creates both components from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - extracting x and y from a 2D point:
        ///
        /// ```
        /// use nalgebra::Point2;
        ///
        /// let point = Point2::new(3.0, 7.0);
        /// let swizzled = point.xy();
        ///
        /// assert_eq!(swizzled, Point2::new(3.0, 7.0));
        /// ```
        ///
        /// Extracting x and y from a 3D point:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point3d = Point3::new(1.0, 2.0, 3.0);
        /// let point2d = point3d.xy();
        ///
        /// assert_eq!(point2d, Point2::new(1.0, 2.0));
        /// // The z-component (3.0) is discarded
        /// ```
        ///
        /// Practical use case - converting 3D positions to screen coordinates:
        ///
        /// ```
        /// use nalgebra::{Point3, Point2};
        ///
        /// // A 3D point in world space
        /// let world_pos = Point3::new(10.0, 20.0, 5.0);
        ///
        /// // Extract 2D screen position (ignoring depth)
        /// let screen_pos = world_pos.xy();
        ///
        /// assert_eq!(screen_pos, Point2::new(10.0, 20.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`yx()`](Self::yx) - swaps x and y components
        /// - [`xz()`](Self::xz) - extracts x and z components
        /// - [`yz()`](Self::yz) - extracts y and z components
        /// - [`xyz()`](Self::xyz) - extracts all three components as 3D point
        #[inline]
        #[must_use]
        pub fn xy(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    (@method yx, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point by swapping the x and y components.
        ///
        /// This is a **swizzle** operation that reverses the order of the first two coordinates.
        /// The method name `yx` indicates that the output's first component comes from y,
        /// and the second component comes from x.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.xx()` creates both components from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - swapping x and y:
        ///
        /// ```
        /// use nalgebra::Point2;
        ///
        /// let point = Point2::new(3.0, 7.0);
        /// let swizzled = point.yx();
        ///
        /// assert_eq!(swizzled, Point2::new(7.0, 3.0));
        /// ```
        ///
        /// Practical use case - converting between coordinate systems:
        ///
        /// ```
        /// use nalgebra::Point2;
        ///
        /// // In some graphics APIs, y points down; in others, y points up
        /// // Swizzling can help convert between conventions
        /// let texture_coords = Point2::new(0.5, 0.8);
        ///
        /// // Swap to row-column indexing
        /// let matrix_index = texture_coords.yx();
        ///
        /// assert_eq!(matrix_index, Point2::new(0.8, 0.5));
        /// ```
        ///
        /// Transposing 2D data:
        ///
        /// ```
        /// use nalgebra::Point2;
        ///
        /// let dimensions = Point2::new(1920, 1080);  // width, height
        /// let transposed = dimensions.yx();           // height, width
        ///
        /// assert_eq!(transposed, Point2::new(1080, 1920));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xy()`](Self::xy) - preserves original x and y order
        /// - [`zyx()`](Self::zyx) - reverses all three components in 3D
        /// - [`yy()`](Self::yy) - duplicates the y-component
        /// - [`xx()`](Self::xx) - duplicates the x-component
        #[inline]
        #[must_use]
        pub fn yx(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    (@method yy, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point by duplicating the y-component.
        ///
        /// This is a **swizzle** operation that creates both output components from the
        /// y-axis (index 1). The method name `yy` indicates that both components come
        /// from the y coordinate.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.yy()` creates both components from y
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - duplicating the y-component:
        ///
        /// ```
        /// use nalgebra::Point2;
        ///
        /// let point = Point2::new(5.0, 3.0);
        /// let swizzled = point.yy();
        ///
        /// assert_eq!(swizzled, Point2::new(3.0, 3.0));
        /// ```
        ///
        /// Practical use case - extracting height as uniform scaling:
        ///
        /// ```
        /// use nalgebra::Point2;
        ///
        /// let dimensions = Point2::new(100.0, 50.0);  // width, height
        /// // Create a square with size equal to height
        /// let square = dimensions.yy();
        ///
        /// assert_eq!(square, Point2::new(50.0, 50.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xx()`](Self::xx) - duplicates the x-component
        /// - [`xy()`](Self::xy) - preserves original x and y components
        /// - [`yx()`](Self::yx) - swaps x and y components
        /// - [`yyy()`](Self::yyy) - creates a 3D point from duplicated y
        #[inline]
        #[must_use]
        pub fn yy(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    (@method xz, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point from the x and z components.
        ///
        /// This is a **swizzle** operation that extracts the x (index 0) and z (index 2)
        /// components, skipping the y component. This is useful for working with horizontal
        /// planes or converting 3D coordinates to 2D.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.xx()` creates both components from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - extracting x and z from a 3D point:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point3d = Point3::new(1.0, 2.0, 3.0);
        /// let point2d = point3d.xz();
        ///
        /// assert_eq!(point2d.x, 1.0);
        /// assert_eq!(point2d.y, 3.0);
        /// // Note: the result's y-component contains the original z value
        /// ```
        ///
        /// Practical use case - projecting 3D points onto the xz-plane (ground plane):
        ///
        /// ```
        /// use nalgebra::{Point3, Point2};
        ///
        /// // A 3D position in world space (y is up)
        /// let world_pos = Point3::new(10.0, 5.0, 8.0);
        ///
        /// // Project onto the ground plane (xz-plane)
        /// let ground_pos = world_pos.xz();
        ///
        /// assert_eq!(ground_pos, Point2::new(10.0, 8.0));
        /// // The height (y=5.0) is discarded
        /// ```
        ///
        /// Working with top-down 2D maps from 3D space:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let character_pos = Point3::new(25.0, 12.0, 30.0);
        /// let map_position = character_pos.xz();
        ///
        /// // map_position now represents the character's position on a 2D map
        /// assert_eq!(map_position.x, 25.0);
        /// assert_eq!(map_position.y, 30.0);
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xy()`](Self::xy) - extracts x and y components
        /// - [`yz()`](Self::yz) - extracts y and z components
        /// - [`zx()`](Self::zx) - extracts z and x (reverse order)
        /// - [`xyz()`](Self::xyz) - extracts all three components
        #[inline]
        #[must_use]
        pub fn xz(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    (@method yz, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point from the y and z components.
        ///
        /// This is a **swizzle** operation that extracts the y (index 1) and z (index 2)
        /// components, discarding the x component. This is useful for side-view projections
        /// or extracting vertical slices from 3D space.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.xx()` creates both components from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - extracting y and z from a 3D point:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point3d = Point3::new(1.0, 2.0, 3.0);
        /// let point2d = point3d.yz();
        ///
        /// assert_eq!(point2d.x, 2.0);
        /// assert_eq!(point2d.y, 3.0);
        /// // Note: the result's x-component contains the original y value
        /// ```
        ///
        /// Practical use case - creating a side-view projection:
        ///
        /// ```
        /// use nalgebra::{Point3, Point2};
        ///
        /// // A 3D position (x=forward, y=up, z=right in some coordinate systems)
        /// let world_pos = Point3::new(10.0, 5.0, 8.0);
        ///
        /// // Project onto the yz-plane for a side view
        /// let side_view = world_pos.yz();
        ///
        /// assert_eq!(side_view, Point2::new(5.0, 8.0));
        /// // The forward position (x=10.0) is discarded
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xy()`](Self::xy) - extracts x and y components
        /// - [`xz()`](Self::xz) - extracts x and z components
        /// - [`zy()`](Self::zy) - extracts z and y (reverse order)
        /// - [`xyz()`](Self::xyz) - extracts all three components
        #[inline]
        #[must_use]
        pub fn yz(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    (@method zx, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point from the z and x components.
        ///
        /// This is a **swizzle** operation that extracts the z (index 2) and x (index 0)
        /// components in that order, skipping the y component. This reverses the horizontal
        /// axes compared to [`xz()`](Self::xz).
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.xx()` creates both components from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - extracting z and x in reverse order:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point3d = Point3::new(1.0, 2.0, 3.0);
        /// let point2d = point3d.zx();
        ///
        /// assert_eq!(point2d.x, 3.0);
        /// assert_eq!(point2d.y, 1.0);
        /// ```
        ///
        /// Comparing with xz:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(5.0, 10.0, 15.0);
        ///
        /// let xz = point.xz();
        /// let zx = point.zx();
        ///
        /// assert_eq!(xz.x, 5.0);
        /// assert_eq!(xz.y, 15.0);
        ///
        /// assert_eq!(zx.x, 15.0);
        /// assert_eq!(zx.y, 5.0);
        /// // zx is the reverse of xz
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xz()`](Self::xz) - extracts x and z in normal order
        /// - [`zy()`](Self::zy) - extracts z and y components
        /// - [`zyx()`](Self::zyx) - reverses all three components
        #[inline]
        #[must_use]
        pub fn zx(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    (@method zy, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point from the z and y components.
        ///
        /// This is a **swizzle** operation that extracts the z (index 2) and y (index 1)
        /// components in that order, discarding the x component. This is the reverse order
        /// of [`yz()`](Self::yz).
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.xx()` creates both components from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - extracting z and y in reverse order:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point3d = Point3::new(1.0, 2.0, 3.0);
        /// let point2d = point3d.zy();
        ///
        /// assert_eq!(point2d.x, 3.0);
        /// assert_eq!(point2d.y, 2.0);
        /// ```
        ///
        /// Comparing with yz:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(5.0, 10.0, 15.0);
        ///
        /// let yz = point.yz();
        /// let zy = point.zy();
        ///
        /// assert_eq!(yz.x, 10.0);
        /// assert_eq!(yz.y, 15.0);
        ///
        /// assert_eq!(zy.x, 15.0);
        /// assert_eq!(zy.y, 10.0);
        /// // zy is the reverse of yz
        /// ```
        ///
        /// # See Also
        ///
        /// - [`yz()`](Self::yz) - extracts y and z in normal order
        /// - [`zx()`](Self::zx) - extracts z and x components
        /// - [`zyx()`](Self::zyx) - reverses all three components
        #[inline]
        #[must_use]
        pub fn zy(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    (@method zz, Point2, $x:expr, $y:expr) => {
        /// Creates a new 2D point by duplicating the z-component.
        ///
        /// This is a **swizzle** operation that creates both output components from the
        /// z-axis (index 2). The method name `zz` indicates that both components come
        /// from the z coordinate.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.yx()` swaps x and y coordinates
        /// - **Duplicate** components: `point.zz()` creates both components from z
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - duplicating the z-component:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(5.0, 3.0, 7.0);
        /// let swizzled = point.zz();
        ///
        /// assert_eq!(swizzled.x, 7.0);
        /// assert_eq!(swizzled.y, 7.0);
        /// ```
        ///
        /// Practical use case - extracting depth as uniform value:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let position = Point3::new(10.0, 20.0, 5.0);
        /// // Create a 2D point with both components equal to the depth
        /// let depth_pair = position.zz();
        ///
        /// assert_eq!(depth_pair.x, 5.0);
        /// assert_eq!(depth_pair.y, 5.0);
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xx()`](Self::xx) - duplicates the x-component
        /// - [`yy()`](Self::yy) - duplicates the y-component
        /// - [`zzz()`](Self::zzz) - creates a 3D point from duplicated z
        #[inline]
        #[must_use]
        pub fn zz(&self) -> Point2<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point2::new(self[$x].clone(), self[$y].clone())
        }
    };

    // Three-component swizzles - xxx
    (@method xxx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point by triplicating the x-component.
        ///
        /// This is a **swizzle** operation that creates all three output components from the
        /// x-axis (index 0). The method name `xxx` indicates that all components come from x.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxx()` creates all components from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - triplicating the x-component:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point = Point2::new(5.0, 3.0);
        /// let swizzled = point.xxx();
        ///
        /// assert_eq!(swizzled, Point3::new(5.0, 5.0, 5.0));
        /// ```
        ///
        /// Practical use case - creating uniform 3D scaling from a single value:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let scale_factor = Point2::new(2.0, 999.0);
        /// // Extract just the x-component as a uniform 3D scale
        /// let uniform_scale_3d = scale_factor.xxx();
        ///
        /// assert_eq!(uniform_scale_3d, Point3::new(2.0, 2.0, 2.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xx()`](Self::xx) - duplicates x in 2D
        /// - [`yyy()`](Self::yyy) - triplicates the y-component
        /// - [`zzz()`](Self::zzz) - triplicates the z-component
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        #[inline]
        #[must_use]
        pub fn xxx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U0, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method xxy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (x, x, y).
        ///
        /// This is a **swizzle** operation that duplicates the x-component for the first
        /// two positions and places y in the third position.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxy()` repeats x twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point = Point2::new(3.0, 7.0);
        /// let swizzled = point.xxy();
        ///
        /// assert_eq!(swizzled, Point3::new(3.0, 3.0, 7.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xyx()`](Self::xyx) - pattern (x, y, x)
        /// - [`xyy()`](Self::xyy) - pattern (x, y, y)
        #[inline]
        #[must_use]
        pub fn xxy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method xyx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (x, y, x).
        ///
        /// This is a **swizzle** operation that places x in the first and third positions,
        /// with y in the middle.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xyx()` repeats x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point = Point2::new(3.0, 7.0);
        /// let swizzled = point.xyx();
        ///
        /// assert_eq!(swizzled, Point3::new(3.0, 7.0, 3.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xxy()`](Self::xxy) - pattern (x, x, y)
        /// - [`yxy()`](Self::yxy) - pattern (y, x, y)
        #[inline]
        #[must_use]
        pub fn xyx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method xyy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (x, y, y).
        ///
        /// This is a **swizzle** operation that places x in the first position and
        /// duplicates y for the last two positions.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xyy()` repeats y twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point = Point2::new(3.0, 7.0);
        /// let swizzled = point.xyy();
        ///
        /// assert_eq!(swizzled, Point3::new(3.0, 7.0, 7.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xxy()`](Self::xxy) - pattern (x, x, y)
        /// - [`yyx()`](Self::yyx) - pattern (y, y, x)
        #[inline]
        #[must_use]
        pub fn xyy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yxx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (y, x, x).
        ///
        /// This is a **swizzle** operation that places y in the first position and
        /// duplicates x for the last two positions.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.yxx()` repeats x twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point = Point2::new(3.0, 7.0);
        /// let swizzled = point.yxx();
        ///
        /// assert_eq!(swizzled, Point3::new(7.0, 3.0, 3.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xyy()`](Self::xyy) - pattern (x, y, y)
        /// - [`xxy()`](Self::xxy) - pattern (x, x, y)
        #[inline]
        #[must_use]
        pub fn yxx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yxy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (y, x, y).
        ///
        /// This is a **swizzle** operation that duplicates y in the first and third positions,
        /// with x in the middle.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.yxy()` repeats y
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point = Point2::new(3.0, 7.0);
        /// let swizzled = point.yxy();
        ///
        /// assert_eq!(swizzled, Point3::new(7.0, 3.0, 7.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xyx()`](Self::xyx) - pattern (x, y, x)
        /// - [`yyy()`](Self::yyy) - triplicates y
        #[inline]
        #[must_use]
        pub fn yxy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yyx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (y, y, x).
        ///
        /// This is a **swizzle** operation that duplicates y for the first two positions
        /// and places x in the third position.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.yyx()` repeats y twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point = Point2::new(3.0, 7.0);
        /// let swizzled = point.yyx();
        ///
        /// assert_eq!(swizzled, Point3::new(7.0, 7.0, 3.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xyy()`](Self::xyy) - pattern (x, y, y)
        /// - [`yyy()`](Self::yyy) - triplicates y
        #[inline]
        #[must_use]
        pub fn yyx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yyy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point by triplicating the y-component.
        ///
        /// This is a **swizzle** operation that creates all three output components from the
        /// y-axis (index 1). The method name `yyy` indicates that all components come from y.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.yyy()` creates all components from y
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - triplicating the y-component:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point = Point2::new(5.0, 3.0);
        /// let swizzled = point.yyy();
        ///
        /// assert_eq!(swizzled, Point3::new(3.0, 3.0, 3.0));
        /// ```
        ///
        /// Practical use case - creating uniform 3D value from height:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let dimensions = Point2::new(100.0, 50.0);  // width, height
        /// // Create a uniform 3D value from the height
        /// let uniform_height = dimensions.yyy();
        ///
        /// assert_eq!(uniform_height, Point3::new(50.0, 50.0, 50.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`yy()`](Self::yy) - duplicates y in 2D
        /// - [`xxx()`](Self::xxx) - triplicates the x-component
        /// - [`zzz()`](Self::zzz) - triplicates the z-component
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        #[inline]
        #[must_use]
        pub fn yyy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U1, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method xxz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (x, x, z).
        ///
        /// This is a **swizzle** operation that duplicates x for the first two positions
        /// and places z in the third position.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxz()` repeats x twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.xxz();
        ///
        /// assert_eq!(swizzled, Point3::new(3.0, 3.0, 9.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xxy()`](Self::xxy) - pattern (x, x, y)
        /// - [`xzz()`](Self::xzz) - pattern (x, z, z)
        #[inline]
        #[must_use]
        pub fn xxz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method xyz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from the x, y, and z components in order.
        ///
        /// This is a **swizzle** operation that preserves the original component order.
        /// For 3D points, this effectively creates a copy. For higher-dimensional points,
        /// this extracts just the first three components.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxx()` creates all from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - preserving order in a 3D point:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(1.0, 2.0, 3.0);
        /// let swizzled = point.xyz();
        ///
        /// assert_eq!(swizzled, Point3::new(1.0, 2.0, 3.0));
        /// ```
        ///
        /// Extracting 3D position from higher dimensions:
        ///
        /// ```
        /// use nalgebra::{Point4, Point3};
        ///
        /// // A 4D point (e.g., homogeneous coordinates)
        /// let point4d = Point4::new(10.0, 20.0, 30.0, 1.0);
        /// let point3d = point4d.xyz();
        ///
        /// assert_eq!(point3d, Point3::new(10.0, 20.0, 30.0));
        /// // The w-component (1.0) is discarded
        /// ```
        ///
        /// Practical use case - converting from 2D to 3D by adding a z-component:
        ///
        /// ```
        /// use nalgebra::{Point2, Point3};
        ///
        /// let point2d = Point2::new(5.0, 10.0);
        /// // This won't compile - xyz() requires at least 3 components
        /// // let point3d = point2d.xyz();  // Error!
        ///
        /// // Instead, construct directly or use other methods
        /// let point3d = Point3::new(point2d.x, point2d.y, 0.0);
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xy()`](Self::xy) - extracts just x and y as 2D
        /// - [`xz()`](Self::xz) - extracts x and z as 2D
        /// - [`yz()`](Self::yz) - extracts y and z as 2D
        /// - [`zyx()`](Self::zyx) - reverses all three components
        /// - [`xzy()`](Self::xzy) - swaps y and z
        /// - [`yxz()`](Self::yxz) - cycles components
        #[inline]
        #[must_use]
        pub fn xyz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method xzx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (x, z, x).
        ///
        /// This is a **swizzle** operation that duplicates x in the first and third positions,
        /// with z in the middle.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xzx()` repeats x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.xzx();
        ///
        /// assert_eq!(swizzled, Point3::new(3.0, 9.0, 3.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xyx()`](Self::xyx) - pattern (x, y, x)
        /// - [`zxz()`](Self::zxz) - pattern (z, x, z)
        #[inline]
        #[must_use]
        pub fn xzx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method xzy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (x, z, y).
        ///
        /// This is a **swizzle** operation that swaps the y and z components while keeping
        /// x in the first position. This is useful for converting between different 3D
        /// coordinate system conventions.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxx()` creates all from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - swapping y and z:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(1.0, 2.0, 3.0);
        /// let swizzled = point.xzy();
        ///
        /// assert_eq!(swizzled, Point3::new(1.0, 3.0, 2.0));
        /// ```
        ///
        /// Practical use case - converting between coordinate systems:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// // In some 3D systems, y is up; in others, z is up
        /// // Convert from Y-up to Z-up coordinate system
        /// let y_up = Point3::new(5.0, 10.0, 8.0);  // x, height, z
        /// let z_up = y_up.xzy();                    // x, z, height
        ///
        /// assert_eq!(z_up, Point3::new(5.0, 8.0, 10.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xzy()`](Self::xzy) - swaps y and z
        /// - [`yxz()`](Self::yxz) - swaps x and y
        /// - [`zyx()`](Self::zyx) - reverses all components
        #[inline]
        #[must_use]
        pub fn xzy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method xzz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (x, z, z).
        ///
        /// This is a **swizzle** operation that places x first and duplicates z for
        /// the last two positions.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xzz()` repeats z twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.xzz();
        ///
        /// assert_eq!(swizzled, Point3::new(3.0, 9.0, 9.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xxz()`](Self::xxz) - pattern (x, x, z)
        /// - [`zzz()`](Self::zzz) - triplicates z
        #[inline]
        #[must_use]
        pub fn xzz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yxz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (y, x, z).
        ///
        /// This is a **swizzle** operation that swaps the x and y components while keeping
        /// z in the third position.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxx()` creates all from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - swapping x and y:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(1.0, 2.0, 3.0);
        /// let swizzled = point.yxz();
        ///
        /// assert_eq!(swizzled, Point3::new(2.0, 1.0, 3.0));
        /// ```
        ///
        /// Practical use case - converting between row-major and column-major indexing:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let position = Point3::new(5.0, 10.0, 2.0);
        /// let reordered = position.yxz();
        ///
        /// assert_eq!(reordered, Point3::new(10.0, 5.0, 2.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xzy()`](Self::xzy) - swaps y and z
        /// - [`zxy()`](Self::zxy) - rotates components left
        /// - [`zyx()`](Self::zyx) - reverses all components
        #[inline]
        #[must_use]
        pub fn yxz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yyz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (y, y, z).
        ///
        /// This is a **swizzle** operation that duplicates y for the first two positions
        /// and places z in the third position.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.yyz()` repeats y twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.yyz();
        ///
        /// assert_eq!(swizzled, Point3::new(7.0, 7.0, 9.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yyy()`](Self::yyy) - triplicates y
        /// - [`yzz()`](Self::yzz) - pattern (y, z, z)
        #[inline]
        #[must_use]
        pub fn yyz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yzx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (y, z, x).
        ///
        /// This is a **swizzle** operation that performs a cyclic rotation of components:
        /// y moves to first position, z to second, and x to third. This is useful for
        /// rotating through coordinate axes.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxx()` creates all from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - cycling components:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(1.0, 2.0, 3.0);
        /// let swizzled = point.yzx();
        ///
        /// assert_eq!(swizzled, Point3::new(2.0, 3.0, 1.0));
        /// ```
        ///
        /// Demonstrating cyclic rotation:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let original = Point3::new(1.0, 2.0, 3.0);
        /// let first = original.yzx();   // (2, 3, 1)
        /// let second = first.yzx();     // (3, 1, 2)
        /// let third = second.yzx();     // (1, 2, 3) - back to original!
        ///
        /// assert_eq!(third, original);
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`zxy()`](Self::zxy) - cycles in reverse direction
        /// - [`yxz()`](Self::yxz) - swaps x and y
        /// - [`xzy()`](Self::xzy) - swaps y and z
        #[inline]
        #[must_use]
        pub fn yzx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yzy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (y, z, y).
        ///
        /// This is a **swizzle** operation that duplicates y in the first and third positions,
        /// with z in the middle.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.yzy()` repeats y
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.yzy();
        ///
        /// assert_eq!(swizzled, Point3::new(7.0, 9.0, 7.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yxy()`](Self::yxy) - pattern (y, x, y)
        /// - [`yyy()`](Self::yyy) - triplicates y
        #[inline]
        #[must_use]
        pub fn yzy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method yzz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (y, z, z).
        ///
        /// This is a **swizzle** operation that places y first and duplicates z for
        /// the last two positions.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.yzz()` repeats z twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.yzz();
        ///
        /// assert_eq!(swizzled, Point3::new(7.0, 9.0, 9.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yyz()`](Self::yyz) - pattern (y, y, z)
        /// - [`zzz()`](Self::zzz) - triplicates z
        #[inline]
        #[must_use]
        pub fn yzz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zxx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (z, x, x).
        ///
        /// This is a **swizzle** operation that places z first and duplicates x for
        /// the last two positions.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.zxx()` repeats x twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.zxx();
        ///
        /// assert_eq!(swizzled, Point3::new(9.0, 3.0, 3.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yxx()`](Self::yxx) - pattern (y, x, x)
        /// - [`zzz()`](Self::zzz) - triplicates z
        #[inline]
        #[must_use]
        pub fn zxx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zxy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (z, x, y).
        ///
        /// This is a **swizzle** operation that performs a cyclic rotation of components
        /// in reverse: z moves to first position, x to second, and y to third.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxx()` creates all from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - cycling components in reverse:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(1.0, 2.0, 3.0);
        /// let swizzled = point.zxy();
        ///
        /// assert_eq!(swizzled, Point3::new(3.0, 1.0, 2.0));
        /// ```
        ///
        /// Demonstrating reverse cyclic rotation:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let original = Point3::new(1.0, 2.0, 3.0);
        /// let first = original.zxy();   // (3, 1, 2)
        /// let second = first.zxy();     // (2, 3, 1)
        /// let third = second.zxy();     // (1, 2, 3) - back to original!
        ///
        /// assert_eq!(third, original);
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yzx()`](Self::yzx) - cycles in forward direction
        /// - [`zyx()`](Self::zyx) - reverses all components
        #[inline]
        #[must_use]
        pub fn zxy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zxz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (z, x, z).
        ///
        /// This is a **swizzle** operation that duplicates z in the first and third positions,
        /// with x in the middle.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.zxz()` repeats z
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.zxz();
        ///
        /// assert_eq!(swizzled, Point3::new(9.0, 3.0, 9.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xzx()`](Self::xzx) - pattern (x, z, x)
        /// - [`zzz()`](Self::zzz) - triplicates z
        #[inline]
        #[must_use]
        pub fn zxz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zyx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point by reversing all three components.
        ///
        /// This is a **swizzle** operation that completely reverses the order of components:
        /// z becomes first, y stays in the middle, and x becomes last. This is useful for
        /// flipping coordinate systems or reversing spatial orderings.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.xxx()` creates all from x
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - reversing all components:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(1.0, 2.0, 3.0);
        /// let reversed = point.zyx();
        ///
        /// assert_eq!(reversed, Point3::new(3.0, 2.0, 1.0));
        /// ```
        ///
        /// Double reversal returns to original:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let original = Point3::new(5.0, 10.0, 15.0);
        /// let reversed = original.zyx();
        /// let back = reversed.zyx();
        ///
        /// assert_eq!(back, original);
        /// ```
        ///
        /// Practical use case - converting between little-endian and big-endian style ordering:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let rgb = Point3::new(255.0, 128.0, 64.0);  // Red, Green, Blue
        /// let bgr = rgb.zyx();                         // Blue, Green, Red
        ///
        /// assert_eq!(bgr, Point3::new(64.0, 128.0, 255.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yx()`](Self::yx) - reverses just x and y
        /// - [`xzy()`](Self::xzy) - swaps y and z
        /// - [`yxz()`](Self::yxz) - swaps x and y
        #[inline]
        #[must_use]
        pub fn zyx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zyy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (z, y, y).
        ///
        /// This is a **swizzle** operation that places z first and duplicates y for
        /// the last two positions.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.zyy()` repeats y twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.zyy();
        ///
        /// assert_eq!(swizzled, Point3::new(9.0, 7.0, 7.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yyz()`](Self::yyz) - pattern (y, y, z)
        /// - [`yyy()`](Self::yyy) - triplicates y
        #[inline]
        #[must_use]
        pub fn zyy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zyz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (z, y, z).
        ///
        /// This is a **swizzle** operation that duplicates z in the first and third positions,
        /// with y in the middle.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.zyz()` repeats z
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.zyz();
        ///
        /// assert_eq!(swizzled, Point3::new(9.0, 7.0, 9.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yzy()`](Self::yzy) - pattern (y, z, y)
        /// - [`zzz()`](Self::zzz) - triplicates z
        #[inline]
        #[must_use]
        pub fn zyz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zzx, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (z, z, x).
        ///
        /// This is a **swizzle** operation that duplicates z for the first two positions
        /// and places x in the third position.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.zzx()` repeats z twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.zzx();
        ///
        /// assert_eq!(swizzled, Point3::new(9.0, 9.0, 3.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`xxz()`](Self::xxz) - pattern (x, x, z)
        /// - [`zzz()`](Self::zzz) - triplicates z
        #[inline]
        #[must_use]
        pub fn zzx(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zzy, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point from components (z, z, y).
        ///
        /// This is a **swizzle** operation that duplicates z for the first two positions
        /// and places y in the third position.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.zzy()` repeats z twice
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(3.0, 7.0, 9.0);
        /// let swizzled = point.zzy();
        ///
        /// assert_eq!(swizzled, Point3::new(9.0, 9.0, 7.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        /// - [`yyz()`](Self::yyz) - pattern (y, y, z)
        /// - [`zzz()`](Self::zzz) - triplicates z
        #[inline]
        #[must_use]
        pub fn zzy(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };

    (@method zzz, Point3, $x:expr, $y:expr, $z:expr) => {
        /// Creates a new 3D point by triplicating the z-component.
        ///
        /// This is a **swizzle** operation that creates all three output components from the
        /// z-axis (index 2). The method name `zzz` indicates that all components come from z.
        ///
        /// # What is Swizzling?
        ///
        /// Swizzling allows you to:
        /// - **Reorder** components: `point.zyx()` reverses all components
        /// - **Duplicate** components: `point.zzz()` creates all components from z
        /// - **Extract** subsets: useful for dimension conversion
        /// - **Select** arbitrary combinations: mix and match any components
        ///
        /// This feature is inspired by shader languages like GLSL and HLSL.
        ///
        /// # Examples
        ///
        /// Basic usage - triplicating the z-component:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let point = Point3::new(5.0, 3.0, 7.0);
        /// let swizzled = point.zzz();
        ///
        /// assert_eq!(swizzled, Point3::new(7.0, 7.0, 7.0));
        /// ```
        ///
        /// Practical use case - creating uniform 3D value from depth:
        ///
        /// ```
        /// use nalgebra::Point3;
        ///
        /// let position = Point3::new(10.0, 20.0, 5.0);
        /// // Create a uniform 3D value from the depth (z) component
        /// let uniform_depth = position.zzz();
        ///
        /// assert_eq!(uniform_depth, Point3::new(5.0, 5.0, 5.0));
        /// ```
        ///
        /// # See Also
        ///
        /// - [`zz()`](Self::zz) - duplicates z in 2D
        /// - [`xxx()`](Self::xxx) - triplicates the x-component
        /// - [`yyy()`](Self::yyy) - triplicates the y-component
        /// - [`xyz()`](Self::xyz) - preserves all components in order
        #[inline]
        #[must_use]
        pub fn zzz(&self) -> Point3<T>
        where
            <Const<D> as ToTypenum>::Typenum: Cmp<typenum::U2, Output = Greater>,
        {
            Point3::new(self[$x].clone(), self[$y].clone(), self[$z].clone())
        }
    };
}

/// # Swizzling
impl<T: Scalar, const D: usize> Point<T, D>
where
    Const<D>: ToTypenum,
{
    impl_swizzle!(
        where U0: xx()  -> Point2[0, 0],
                  xxx() -> Point3[0, 0, 0];

        where U1: xy()  -> Point2[0, 1],
                  yx()  -> Point2[1, 0],
                  yy()  -> Point2[1, 1],
                  xxy() -> Point3[0, 0, 1],
                  xyx() -> Point3[0, 1, 0],
                  xyy() -> Point3[0, 1, 1],
                  yxx() -> Point3[1, 0, 0],
                  yxy() -> Point3[1, 0, 1],
                  yyx() -> Point3[1, 1, 0],
                  yyy() -> Point3[1, 1, 1];

        where U2: xz()  -> Point2[0, 2],
                  yz()  -> Point2[1, 2],
                  zx()  -> Point2[2, 0],
                  zy()  -> Point2[2, 1],
                  zz()  -> Point2[2, 2],
                  xxz() -> Point3[0, 0, 2],
                  xyz() -> Point3[0, 1, 2],
                  xzx() -> Point3[0, 2, 0],
                  xzy() -> Point3[0, 2, 1],
                  xzz() -> Point3[0, 2, 2],
                  yxz() -> Point3[1, 0, 2],
                  yyz() -> Point3[1, 1, 2],
                  yzx() -> Point3[1, 2, 0],
                  yzy() -> Point3[1, 2, 1],
                  yzz() -> Point3[1, 2, 2],
                  zxx() -> Point3[2, 0, 0],
                  zxy() -> Point3[2, 0, 1],
                  zxz() -> Point3[2, 0, 2],
                  zyx() -> Point3[2, 1, 0],
                  zyy() -> Point3[2, 1, 1],
                  zyz() -> Point3[2, 1, 2],
                  zzx() -> Point3[2, 2, 0],
                  zzy() -> Point3[2, 2, 1],
                  zzz() -> Point3[2, 2, 2];
    );
}
