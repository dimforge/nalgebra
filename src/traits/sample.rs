/// Traits of vectors able to sample a sphere. The number of sample must be sufficient to
/// approximate a sphere using support mapping functions.
pub trait UniformSphereSample {
    /// Iterate throught the samples.
    fn sample(&fn(&'static Self));

    /// Gets the list of all samples.
    fn sample_list() -> &[Self];
}
