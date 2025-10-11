//! This module provides the matrix exponential (pow) function to square matrices.

use crate::{
    DefaultAllocator, DimMin, Matrix, OMatrix, Scalar,
    allocator::Allocator,
    storage::{Storage, StorageMut},
};
use num::{One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign};

impl<T, D, S> Matrix<T, D, D, S>
where
    T: Scalar + Zero + One + ClosedAddAssign + ClosedMulAssign,
    D: DimMin<D, Output = D>,
    S: StorageMut<T, D, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Raises this matrix to an integral power `exp` in-place.
    ///
    /// # What is Matrix Power?
    ///
    /// Matrix power is the result of multiplying a square matrix by itself a certain number of times.
    /// For a matrix **A** and a non-negative integer **n**:
    /// - **A**^0 = **I** (identity matrix)
    /// - **A**^1 = **A**
    /// - **A**^2 = **A** × **A**
    /// - **A**^3 = **A** × **A** × **A**
    /// - And so on...
    ///
    /// Matrix powers are useful in many applications including:
    /// - **Markov chains**: Computing state transitions over multiple time steps
    /// - **Graph theory**: Finding paths of length n between vertices
    /// - **Dynamical systems**: Analyzing system evolution over discrete time steps
    /// - **Differential equations**: Numerical solutions using matrix exponentials
    ///
    /// # Arguments
    ///
    /// * `exp` - A non-negative integer exponent (u32). The matrix will be raised to this power.
    ///
    /// # Performance
    ///
    /// This method uses exponentiation by squaring (also known as binary exponentiation),
    /// which computes the result in O(log n) matrix multiplications instead of O(n).
    /// This makes it efficient even for large exponents.
    ///
    /// The computation is done in-place, modifying `self` directly to save memory allocations.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let mut m = Matrix2::new(
    ///     2.0, 1.0,
    ///     0.0, 2.0
    /// );
    ///
    /// // Raise to power 3: m^3 = m * m * m
    /// m.pow_mut(3);
    ///
    /// let expected = Matrix2::new(
    ///     8.0, 12.0,
    ///     0.0, 8.0
    /// );
    ///
    /// assert_eq!(m, expected);
    /// ```
    ///
    /// ## Power of Zero (Identity)
    ///
    /// Any matrix raised to the power of 0 yields the identity matrix:
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let mut m = Matrix3::new(
    ///     5.0, 2.0, 3.0,
    ///     1.0, 4.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// m.pow_mut(0);
    ///
    /// // Result is the identity matrix
    /// assert_eq!(m, Matrix3::identity());
    /// ```
    ///
    /// ## Power of One (No Change)
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let original = Matrix2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0
    /// );
    /// let mut m = original.clone();
    ///
    /// m.pow_mut(1);
    ///
    /// assert_eq!(m, original);
    /// ```
    ///
    /// ## Application: Markov Chain State Transitions
    ///
    /// In a Markov chain, the transition matrix raised to power n gives
    /// the probability of transitioning between states in exactly n steps.
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Simple weather model: [Sunny, Rainy]
    /// // If sunny today: 80% chance sunny tomorrow, 20% chance rainy
    /// // If rainy today: 40% chance sunny tomorrow, 60% chance rainy
    /// let mut transition = Matrix2::new(
    ///     0.8, 0.4,  // Column 0: transitions from Sunny
    ///     0.2, 0.6   // Column 1: transitions from Rainy
    /// );
    ///
    /// // What are the probabilities after 3 days?
    /// transition.pow_mut(3);
    ///
    /// // transition[0][0] is now probability: Sunny -> (3 days) -> Sunny
    /// // transition[1][0] is now probability: Sunny -> (3 days) -> Rainy
    /// assert!((transition[(0, 0)] - 0.688_f64).abs() < 1e-10);
    /// assert!((transition[(1, 0)] - 0.312_f64).abs() < 1e-10);
    /// ```
    ///
    /// ## Application: Graph Paths
    ///
    /// For a graph adjacency matrix, A^n[i][j] counts the number of paths
    /// of length exactly n from vertex i to vertex j.
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Graph with 3 vertices
    /// // Edges: 0->1, 1->2, 2->0, 0->2
    /// let mut adj = Matrix3::new(
    ///     0, 1, 1,  // vertex 0 connects to 1 and 2
    ///     0, 0, 1,  // vertex 1 connects to 2
    ///     1, 0, 0   // vertex 2 connects to 0
    /// );
    ///
    /// // Count paths of length 2
    /// adj.pow_mut(2);
    ///
    /// // adj[i][j] now contains number of 2-step paths from i to j
    /// assert_eq!(adj[(0, 0)], 1);  // One path: 0->1->2->0 or 0->2->0
    /// ```
    ///
    /// ## Application: Fibonacci Sequence
    ///
    /// The Fibonacci sequence can be computed using matrix powers:
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Fibonacci matrix: [[1, 1], [1, 0]]
    /// let mut fib_matrix = Matrix2::new(
    ///     1, 1,
    ///     1, 0
    /// );
    ///
    /// // Computing F(7) = 13
    /// let n = 7;
    /// fib_matrix.pow_mut(n - 1);
    ///
    /// // The (0,0) element is now F(n)
    /// assert_eq!(fib_matrix[(0, 0)], 13);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`pow`](Self::pow) - Returns a new matrix raised to a power (non-mutating version)
    /// - [`exp`](crate::OMatrix::exp) - Computes the matrix exponential e^A (for continuous-time systems)
    /// - [`try_inverse`](Self::try_inverse) - Computes the matrix inverse (equivalent to A^(-1))
    /// - [`transpose`](Self::transpose) - Computes the matrix transpose
    pub fn pow_mut(&mut self, mut exp: u32) {
        // A matrix raised to the zeroth power is just the identity.
        if exp == 0 {
            self.fill_with_identity();
        } else if exp > 1 {
            // We use the buffer to hold the result of multiplier^2, thus avoiding
            // extra allocations.
            let mut x = self.clone_owned();
            let mut workspace = self.clone_owned();

            if exp % 2 == 0 {
                self.fill_with_identity();
            } else {
                // Avoid an useless multiplication by the identity
                // if the exponent is odd.
                exp -= 1;
            }

            // Exponentiation by squares.
            loop {
                if exp % 2 == 1 {
                    self.mul_to(&x, &mut workspace);
                    self.copy_from(&workspace);
                }

                exp /= 2;

                if exp == 0 {
                    break;
                }

                x.mul_to(&x, &mut workspace);
                x.copy_from(&workspace);
            }
        }
    }
}

impl<T, D, S: Storage<T, D, D>> Matrix<T, D, D, S>
where
    T: Scalar + Zero + One + ClosedAddAssign + ClosedMulAssign,
    D: DimMin<D, Output = D>,
    S: StorageMut<T, D, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Raises this matrix to an integral power `exp` and returns the result.
    ///
    /// This is the non-mutating version of [`pow_mut`](Self::pow_mut). It creates a new
    /// matrix containing the result while leaving the original matrix unchanged.
    ///
    /// # What is Matrix Power?
    ///
    /// Matrix power is the result of multiplying a square matrix by itself a certain number of times.
    /// For a matrix **A** and a non-negative integer **n**:
    /// - **A**^0 = **I** (identity matrix)
    /// - **A**^1 = **A**
    /// - **A**^2 = **A** × **A**
    /// - **A**^3 = **A** × **A** × **A**
    /// - And so on...
    ///
    /// Matrix powers are fundamental in many fields:
    /// - **Markov chains**: Analyzing long-term behavior of stochastic processes
    /// - **Graph theory**: Counting walks and analyzing connectivity
    /// - **Dynamical systems**: Simulating discrete-time evolution
    /// - **Computer graphics**: Applying transformations multiple times
    ///
    /// # Arguments
    ///
    /// * `exp` - A non-negative integer exponent (u32). The matrix will be raised to this power.
    ///
    /// # Returns
    ///
    /// A new matrix equal to `self` raised to the power `exp`.
    ///
    /// # Performance
    ///
    /// This method uses exponentiation by squaring (binary exponentiation), achieving
    /// O(log n) matrix multiplications instead of O(n). This makes it efficient even for
    /// large exponents.
    ///
    /// Unlike [`pow_mut`](Self::pow_mut), this method allocates a new matrix to store the result.
    /// If you don't need to preserve the original matrix, consider using `pow_mut` instead
    /// for better memory efficiency.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     2.0, 1.0,
    ///     0.0, 2.0
    /// );
    ///
    /// // Compute m^3 = m * m * m
    /// let result = m.pow(3);
    ///
    /// let expected = Matrix2::new(
    ///     8.0, 12.0,
    ///     0.0, 8.0
    /// );
    ///
    /// assert_eq!(result, expected);
    /// // Original matrix is unchanged
    /// assert_eq!(m[(0, 0)], 2.0);
    /// ```
    ///
    /// ## Powers of 0 and 1
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// // Any matrix to the power 0 is the identity
    /// assert_eq!(m.pow(0), Matrix3::identity());
    ///
    /// // Any matrix to the power 1 is itself
    /// assert_eq!(m.pow(1), m);
    /// ```
    ///
    /// ## Application: Compound Interest with Multiple Scenarios
    ///
    /// Matrix powers can model how multiple scenarios evolve over time.
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Investment growth matrix (simplified model)
    /// // [Conservative, Aggressive]
    /// let growth = Matrix2::new(
    ///     1.05, 0.02,   // Conservative: 5% growth, small cross-influence
    ///     0.01, 1.10    // Aggressive: 10% growth, small cross-influence
    /// );
    ///
    /// // After 10 periods
    /// let future_state = growth.pow(10);
    ///
    /// // The result shows the compounded growth factors
    /// // future_state[0][0] is the growth of a conservative investment
    /// assert!(future_state[(0, 0)] > 1.5);  // More than 50% growth
    /// assert!(future_state[(1, 1)] > 2.0);  // More than double for aggressive
    /// ```
    ///
    /// ## Application: Population Dynamics
    ///
    /// Leslie matrices model age-structured populations over generations:
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Leslie matrix for a population with 3 age groups
    /// // Row 0: birth rates from each age group
    /// // Subdiagonal: survival rates to next age group
    /// let leslie = Matrix3::new(
    ///     0.0, 1.5, 2.0,  // Fertility: young=0, adult=1.5, old=2.0
    ///     0.5, 0.0, 0.0,  // 50% of young survive to adulthood
    ///     0.0, 0.8, 0.0   // 80% of adults survive to old age
    /// );
    ///
    /// // Project population structure 5 generations ahead
    /// let future = leslie.pow(5);
    ///
    /// // future * initial_population gives population after 5 generations
    /// // The matrix shows how each current age group contributes to
    /// // future age groups
    /// ```
    ///
    /// ## Application: Game Theory - Repeated Games
    ///
    /// In repeated games, the transition matrix raised to a power shows
    /// equilibrium convergence:
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Strategy transition probabilities
    /// // [Cooperate, Defect]
    /// let strategy = Matrix2::new(
    ///     0.9, 0.3,  // If cooperating: 90% stay, 30% switch from defect
    ///     0.1, 0.7   // If defecting: 10% switch, 70% stay
    /// );
    ///
    /// // After many rounds (e.g., 20)
    /// let equilibrium = strategy.pow(20);
    ///
    /// // The columns converge to the same steady-state distribution
    /// // This is the long-run probability of each strategy
    /// ```
    ///
    /// ## Application: Network Distance Matrix
    ///
    /// For directed graphs, powers of the adjacency matrix reveal connectivity:
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Social network: who follows whom
    /// // 1 means i follows j
    /// let follows = Matrix4::new(
    ///     0, 1, 0, 0,  // Person 0 follows person 1
    ///     0, 0, 1, 1,  // Person 1 follows persons 2 and 3
    ///     1, 0, 0, 0,  // Person 2 follows person 0
    ///     0, 1, 0, 0   // Person 3 follows person 1
    /// );
    ///
    /// // 2-step connections (friend of friend)
    /// let two_steps = follows.pow(2);
    ///
    /// // two_steps[i][j] = number of 2-step paths from i to j
    /// // This tells us indirect influence in the network
    /// assert_eq!(two_steps[(0, 2)], 1);  // Person 0 can reach person 2 in 2 steps
    /// ```
    ///
    /// ## Application: Discrete Dynamical Systems
    ///
    /// Iterating a linear transformation models discrete dynamical systems:
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector2};
    ///
    /// // Linear transformation (rotation + scaling)
    /// let transform = Matrix2::new(
    ///     0.8, -0.3,
    ///     0.3,  0.8
    /// );
    ///
    /// // Apply transformation 10 times
    /// let transform_10 = transform.pow(10);
    ///
    /// // Initial state
    /// let initial = Vector2::new(1.0, 0.0);
    ///
    /// // State after 10 iterations
    /// let final_state = transform_10 * initial;
    ///
    /// println!("After 10 iterations: {:?}", final_state);
    /// ```
    ///
    /// ## Comparing with Matrix Exponential
    ///
    /// Matrix power (discrete) vs matrix exponential (continuous):
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let a = Matrix2::new(
    ///     0.0, -1.0,
    ///     1.0,  0.0
    /// );
    ///
    /// // Discrete: A^n (repeated multiplication)
    /// let discrete_result = a.pow(4);
    ///
    /// // Continuous: exp(A) (infinite series)
    /// let continuous_result = a.exp();
    ///
    /// // These are different operations!
    /// // pow(n) is for discrete-time systems
    /// // exp() is for continuous-time systems
    /// ```
    ///
    /// # See Also
    ///
    /// - [`pow_mut`](Self::pow_mut) - In-place version that modifies the matrix directly
    /// - [`exp`](crate::OMatrix::exp) - Computes the matrix exponential e^A (for continuous-time systems)
    /// - [`try_inverse`](Self::try_inverse) - Computes the matrix inverse (conceptually A^(-1))
    /// - [`transpose`](Self::transpose) - Computes the matrix transpose A^T
    #[must_use]
    pub fn pow(&self, exp: u32) -> OMatrix<T, D, D> {
        let mut result = self.clone_owned();
        result.pow_mut(exp);
        result
    }
}
