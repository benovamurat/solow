//! Panic-safety and invariant tests for the continuous distributions.
//!
//! These check that CDFs stay in `[0, 1]`, are monotone non-decreasing, that
//! `cdf + sf == 1`, and that extreme / degenerate arguments produce finite,
//! in-range output rather than panicking. They assert no specific reference
//! values, so they do not affect numerical parity.

use proptest::prelude::*;
use solow_distributions::{chi2_cdf, chi2_sf, f_cdf, f_sf, norm_cdf, norm_sf, t_cdf, t_sf};

// ---------------------------------------------------------------------------
// Explicit edge cases.
// ---------------------------------------------------------------------------

#[test]
fn norm_cdf_extremes_in_range() {
    // Every finite argument — including astronomically large ones whose square
    // overflows to +inf — must yield a finite, in-range value (no NaN).
    for &x in &[
        -1e300,
        -1e154,
        -1e100,
        -1e10,
        -40.0,
        0.0,
        40.0,
        1e10,
        1e100,
        1e154,
        1e300,
        f64::MIN,
        f64::MAX,
    ] {
        let c = norm_cdf(x);
        let s = norm_sf(x);
        assert!(
            c.is_finite() && (0.0..=1.0).contains(&c),
            "norm_cdf({x}) = {c}"
        );
        assert!(
            s.is_finite() && (0.0..=1.0).contains(&s),
            "norm_sf({x}) = {s}"
        );
    }
    // Extreme arguments must saturate to the analytic limit, not NaN.
    assert_eq!(norm_cdf(-1e300), 0.0);
    assert_eq!(norm_cdf(1e300), 1.0);
    assert_eq!(norm_sf(-1e300), 1.0);
    assert_eq!(norm_sf(1e300), 0.0);
}

#[test]
fn t_cdf_extremes_in_range() {
    for &df in &[1.0, 5.0, 1e6] {
        for &x in &[-1e6, -3.0, 0.0, 3.0, 1e6] {
            let c = t_cdf(x, df);
            assert!((0.0..=1.0).contains(&c), "t_cdf({x}, {df}) = {c}");
        }
    }
}

#[test]
fn f_and_chi2_cdf_nonpositive_in_range() {
    // x <= 0 lies below the support; the CDF must be 0, not panic.
    assert_eq!(f_cdf(-1.0, 3.0, 7.0), 0.0);
    assert_eq!(f_cdf(0.0, 3.0, 7.0), 0.0);
    assert_eq!(chi2_cdf(-1.0, 4.0), 0.0);
    assert_eq!(chi2_cdf(0.0, 4.0), 0.0);
}

// ---------------------------------------------------------------------------
// Property-based invariants (modest case counts to keep CI fast).
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn norm_cdf_in_unit_interval(x in -1e6f64..1e6) {
        let c = norm_cdf(x);
        prop_assert!(c.is_finite());
        prop_assert!((0.0..=1.0).contains(&c));
        // cdf + sf == 1.
        prop_assert!((c + norm_sf(x) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn norm_cdf_monotone(a in -50.0f64..50.0, d in 0.0f64..50.0) {
        // Φ is non-decreasing.
        prop_assert!(norm_cdf(a + d) + 1e-12 >= norm_cdf(a));
    }

    #[test]
    fn t_cdf_in_unit_interval(x in -1e4f64..1e4, df in 0.5f64..200.0) {
        let c = t_cdf(x, df);
        prop_assert!(c.is_finite());
        prop_assert!((0.0..=1.0).contains(&c));
        prop_assert!((c + t_sf(x, df) - 1.0).abs() < 1e-8);
    }

    #[test]
    fn t_cdf_monotone(a in -100.0f64..100.0, d in 0.0f64..100.0, df in 0.5f64..200.0) {
        prop_assert!(t_cdf(a + d, df) + 1e-10 >= t_cdf(a, df));
    }

    #[test]
    fn f_cdf_in_unit_interval(x in 0.0f64..1e4, dfn in 0.5f64..100.0, dfd in 0.5f64..100.0) {
        let c = f_cdf(x, dfn, dfd);
        prop_assert!(c.is_finite());
        prop_assert!((0.0..=1.0).contains(&c));
        prop_assert!((c + f_sf(x, dfn, dfd) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn f_cdf_monotone(a in 0.0f64..500.0, d in 0.0f64..500.0, dfn in 0.5f64..100.0, dfd in 0.5f64..100.0) {
        prop_assert!(f_cdf(a + d, dfn, dfd) + 1e-9 >= f_cdf(a, dfn, dfd));
    }

    #[test]
    fn chi2_cdf_in_unit_interval(x in 0.0f64..1e4, df in 0.5f64..200.0) {
        let c = chi2_cdf(x, df);
        prop_assert!(c.is_finite());
        prop_assert!((0.0..=1.0).contains(&c));
        prop_assert!((c + chi2_sf(x, df) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn chi2_cdf_monotone(a in 0.0f64..1000.0, d in 0.0f64..1000.0, df in 0.5f64..200.0) {
        prop_assert!(chi2_cdf(a + d, df) + 1e-9 >= chi2_cdf(a, df));
    }
}
