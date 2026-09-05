//! Two one-sided tests (TOST) of equivalence for two independent samples.
//!
//! [`ttost_ind`] tests the null of *non*-equivalence — that the mean difference
//! `m1 − m2` lies outside the interval `(low, upp)` — against the alternative
//! that it lies inside. It runs an upper one-sided t-test at `low` and a lower
//! one-sided t-test at `upp` and reports the larger of the two p-values.
//!
//! Mirrors the reference `…stats.weightstats.ttost_ind`.

use crate::weightstats::{ttest_ind, Alternative, UseVar};
use ndarray::Array1;
use solow_core::error::Result;

/// Outcome of a TOST equivalence test.
#[derive(Debug, Clone, Copy)]
pub struct TostResult {
    /// p-value of the equivalence test (the larger of the two one-sided
    /// p-values); reject non-equivalence when this is small.
    pub pvalue: f64,
    /// Statistic of the lower-bound (`larger`) one-sided test.
    pub t1: f64,
    /// p-value of the lower-bound one-sided test.
    pub pv1: f64,
    /// Statistic of the upper-bound (`smaller`) one-sided test.
    pub t2: f64,
    /// p-value of the upper-bound one-sided test.
    pub pv2: f64,
}

/// Two one-sided equivalence test for two independent samples.
///
/// `low` and `upp` bound the equivalence interval `low < m1 − m2 < upp`.
/// `usevar` selects the pooled (Student) or unequal-variance (Welch) two-sample
/// t-test. The first one-sided test (`alternative = larger`, null difference
/// `low`) and the second (`alternative = smaller`, null difference `upp`) are
/// taken; the returned `pvalue` is `max(pv1, pv2)`. Mirrors the reference
/// `ttost_ind(x1, x2, low, upp, usevar)`.
pub fn ttost_ind(
    x1: &Array1<f64>,
    x2: &Array1<f64>,
    low: f64,
    upp: f64,
    usevar: UseVar,
) -> Result<TostResult> {
    let tt1 = ttest_ind(x1, x2, Alternative::Larger, usevar, low);
    let tt2 = ttest_ind(x1, x2, Alternative::Smaller, usevar, upp);
    let pvalue = tt1.pvalue.max(tt2.pvalue);
    Ok(TostResult {
        pvalue,
        t1: tt1.statistic,
        pv1: tt1.pvalue,
        t2: tt2.statistic,
        pv2: tt2.pvalue,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tost_runs_pooled_and_unequal() {
        let x1 = array![1.0, 1.2, 0.9, 1.1, 1.05, 0.95, 1.15, 0.85];
        let x2 = array![1.05, 1.1, 1.0, 0.9, 1.2, 1.0, 0.95, 1.1];
        for uv in [UseVar::Pooled, UseVar::Unequal] {
            let r = ttost_ind(&x1, &x2, -0.5, 0.5, uv).unwrap();
            assert!((0.0..=1.0).contains(&r.pvalue));
            assert!((r.pvalue - r.pv1.max(r.pv2)).abs() < 1e-15);
        }
    }
}
