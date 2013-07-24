/// Traits of vectors able to sample a sphere. The number of sample must be sufficient to
/// approximate a sphere using support mapping functions.
pub trait UniformSphereSample
{
  /// Iterate throught the samples.
  pub fn sample(&fn(&'static Self));

  /// Gets the list of all samples.
  pub fn sample_list() -> ~[&'static Self]
  {
    let mut res = ~[];

    do UniformSphereSample::sample::<Self> |s|
    { res.push(s) }

    res
  }
}
