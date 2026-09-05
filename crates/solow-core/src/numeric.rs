//! Numerically stable primitives — compensated summation, dot product, and
//! mean/variance in one pass.
//!
//! Naive floating-point summation loses precision as the running total grows
//! relative to the next term. Two well-established compensation schemes fix
//! this without changing the type or the API meaningfully:
//!
//! * [`KahanSum`] — Kahan (1965) compensated summation. One extra fma-style
//!   step per addition; accurate when successive terms are similar in
//!   magnitude to each other and to the running total.
//! * [`NeumaierSum`] — Neumaier's (1974) variant of Kahan that also handles
//!   the case where the incoming term is *larger* than the running total.
//!   This is the recommended default for mixed-magnitude data (regression
//!   residuals, likelihood contributions, weighted metrics).
//!
//! Both accumulators return the same value on well-behaved input and only
//! diverge from a naive `Iterator::sum` on adversarial or long ill-scaled
//! sequences — which is exactly the regime where the difference matters.
//!
//! Convenience free functions [`kahan_sum`], [`neumaier_sum`],
//! [`compensated_mean`], [`compensated_dot`], and [`compensated_mean_and_var`]
//! wrap the accumulators for one-shot use. These are the building blocks the
//! rest of the Solow stack should reach for when reducing an iterator of
//! floats to a scalar.

/// Kahan (1965) compensated-summation accumulator.
///
/// See the [module-level docs](crate::numeric) for when to prefer this over
/// [`NeumaierSum`].
#[derive(Copy, Clone, Debug, Default)]
pub struct KahanSum {
    sum: f64,
    /// Running compensation for lost low-order bits.
    c: f64,
}

impl KahanSum {
    /// A zero-initialised accumulator.
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    /// Add one term.
    #[inline]
    pub fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    /// Consume the accumulator and return the compensated sum.
    #[inline]
    pub fn finish(self) -> f64 {
        self.sum
    }

    /// Current running total (does not consume the accumulator).
    #[inline]
    pub fn value(&self) -> f64 {
        self.sum
    }
}

/// Neumaier (1974) improved compensated-summation accumulator.
///
/// Behaves like [`KahanSum`] when the running total dominates the incoming
/// term and switches roles when the term dominates, so it stays accurate
/// even when the summed sequence has mixed magnitudes.
#[derive(Copy, Clone, Debug, Default)]
pub struct NeumaierSum {
    sum: f64,
    c: f64,
}

impl NeumaierSum {
    /// A zero-initialised accumulator.
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    /// Add one term.
    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.c += (self.sum - t) + x;
        } else {
            self.c += (x - t) + self.sum;
        }
        self.sum = t;
    }

    /// Consume the accumulator and return the compensated sum
    /// `sum + compensation`.
    #[inline]
    pub fn finish(self) -> f64 {
        self.sum + self.c
    }

    /// Current running total, including the compensation.
    #[inline]
    pub fn value(&self) -> f64 {
        self.sum + self.c
    }
}

// ---------------------------------------------------------------------------
// Free-function conveniences
// ---------------------------------------------------------------------------

/// Kahan-compensated sum of an iterator of `f64`.
pub fn kahan_sum<I: IntoIterator<Item = f64>>(iter: I) -> f64 {
    let mut k = KahanSum::new();
    for x in iter {
        k.add(x);
    }
    k.finish()
}

/// Neumaier-compensated sum of an iterator of `f64`. Prefer this over
/// [`kahan_sum`] for mixed-magnitude data.
pub fn neumaier_sum<I: IntoIterator<Item = f64>>(iter: I) -> f64 {
    let mut n = NeumaierSum::new();
    for x in iter {
        n.add(x);
    }
    n.finish()
}

/// Compensated arithmetic mean of a non-empty iterator of `f64`.
///
/// Returns `f64::NAN` for an empty iterator (there is no defined mean).
pub fn compensated_mean<I: IntoIterator<Item = f64>>(iter: I) -> f64 {
    let mut n = 0usize;
    let mut acc = NeumaierSum::new();
    for x in iter {
        acc.add(x);
        n += 1;
    }
    if n == 0 {
        return f64::NAN;
    }
    acc.finish() / n as f64
}

/// Compensated dot product `Σ xᵢ · yᵢ` of two same-length slices. Panics
/// if the slices have different lengths; the caller is expected to have
/// checked shapes.
pub fn compensated_dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(
        x.len(),
        y.len(),
        "compensated_dot: shape mismatch ({} vs {})",
        x.len(),
        y.len()
    );
    let mut acc = NeumaierSum::new();
    for i in 0..x.len() {
        acc.add(x[i] * y[i]);
    }
    acc.finish()
}

/// Welford one-pass mean and (unbiased) variance of a non-empty iterator.
///
/// Returns `(mean, variance)`. For `n = 1` the variance is `0.0`. Uses the
/// Welford / Chan-Golub-LeVeque recurrence, so it needs a single pass and
/// avoids the catastrophic cancellation of the "sum-of-squares minus
/// mean-squared" formula.
///
/// Returns `(f64::NAN, f64::NAN)` for an empty iterator.
pub fn compensated_mean_and_var<I: IntoIterator<Item = f64>>(iter: I) -> (f64, f64) {
    let mut n = 0usize;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for x in iter {
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    match n {
        0 => (f64::NAN, f64::NAN),
        1 => (mean, 0.0),
        _ => (mean, m2 / (n as f64 - 1.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A classical stress test: `[1.0, 1e100, 1.0, -1e100]` sums to `2.0`
    /// exactly, but naive summation catastrophically loses the `1.0`s.
    /// Neumaier recovers the answer; Kahan does not on this exact sequence
    /// because the first `+1e100` breaks its assumption that terms are
    /// no larger than the running total.
    #[test]
    fn neumaier_beats_naive_on_the_kahan_torture_sequence() {
        let terms = [1.0_f64, 1e100, 1.0, -1e100];
        let naive: f64 = terms.iter().sum();
        assert!((naive - 2.0).abs() > 0.5, "naive sum should be badly wrong");
        assert_eq!(neumaier_sum(terms), 2.0);
    }

    #[test]
    fn compensated_mean_matches_naive_on_well_scaled_data() {
        let xs: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        let naive_mean: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
        assert_eq!(compensated_mean(xs.iter().copied()), naive_mean);
    }

    #[test]
    fn welford_matches_two_pass_variance() {
        let xs = [4.0_f64, 7.0, 13.0, 16.0];
        let (mean, var) = compensated_mean_and_var(xs.iter().copied());
        assert!((mean - 10.0).abs() < 1e-12);
        // Two-pass reference: sample variance with (n - 1) denominator.
        let mu: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
        let ref_var: f64 = xs.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / 3.0;
        assert!((var - ref_var).abs() < 1e-12);
    }

    #[test]
    fn compensated_dot_reduces_to_the_expected_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(compensated_dot(&a, &b), 300.0);
    }

    #[test]
    fn compensated_mean_returns_nan_on_empty_input() {
        assert!(compensated_mean(std::iter::empty::<f64>()).is_nan());
        let (m, v) = compensated_mean_and_var(std::iter::empty::<f64>());
        assert!(m.is_nan() && v.is_nan());
    }
}
