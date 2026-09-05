//! Regression test for the damped Newton step: the discrete-choice estimators must
//! converge on moderately ill-scaled real designs.
//!
//! Uses the canonical "cpunish" capital-punishment Poisson dataset (17 observations,
//! six predictors plus an intercept, design condition number around 1.3e6). With an
//! undamped full-Newton step this overshoots and does not converge; with step damping
//! and a scale-aware convergence test it reaches the certified MLE. The expected
//! coefficients are the maximum-likelihood estimates reported by an authoritative
//! reference, embedded here as constants.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use solow_discrete::Poisson;

#[rustfmt::skip]
const X_ROWS: [[f64; 7]; 17] = [
    [1.0, 34453.0, 16.7, 12.2, 644.0, 1.0, 0.16],
    [1.0, 41534.0, 12.5, 20.0, 351.0, 1.0, 0.27],
    [1.0, 35802.0, 10.6, 11.2, 591.0, 0.0, 0.21],
    [1.0, 26954.0, 18.4, 16.1, 524.0, 1.0, 0.16],
    [1.0, 31468.0, 14.8, 25.9, 565.0, 1.0, 0.19],
    [1.0, 32552.0, 18.8,  3.5, 632.0, 0.0, 0.25],
    [1.0, 40873.0, 11.6, 15.3, 886.0, 0.0, 0.25],
    [1.0, 34861.0, 13.1, 30.1, 997.0, 1.0, 0.21],
    [1.0, 42562.0,  9.4,  4.3, 405.0, 0.0, 0.31],
    [1.0, 31900.0, 14.3, 15.4, 1051.0, 1.0, 0.24],
    [1.0, 37421.0,  8.2,  8.2, 537.0, 0.0, 0.19],
    [1.0, 33305.0, 16.4,  7.2, 321.0, 0.0, 0.16],
    [1.0, 32108.0, 18.4, 32.1, 929.0, 1.0, 0.18],
    [1.0, 45844.0,  9.3, 27.4, 931.0, 0.0, 0.29],
    [1.0, 34743.0, 10.0,  4.0, 435.0, 0.0, 0.24],
    [1.0, 29709.0, 15.2,  7.7, 597.0, 0.0, 0.21],
    [1.0, 36777.0, 11.7,  1.8, 463.0, 0.0, 0.25],
];

const Y: [f64; 17] = [
    37.0, 9.0, 6.0, 4.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];

// Reference Poisson MLE (Newton) for cpunish.
const EXPECTED_PARAMS: [f64; 7] = [
    -4.770212977499e+00,
    2.566657572812e-04,
    7.367587968841e-02,
    -9.248670213461e-02,
    1.887376557128e-04,
    2.310827700090e+00,
    -1.912765882586e+01,
];

#[test]
fn poisson_converges_on_ill_scaled_cpunish_design() {
    let y = Array1::from_vec(Y.to_vec());
    let mut x = Array2::<f64>::zeros((17, 7));
    for (i, row) in X_ROWS.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            x[[i, j]] = v;
        }
    }

    let res = Poisson::new(y, x).unwrap().fit().unwrap();

    // The damped Newton step must actually converge on this cond ~1.3e6 design.
    assert!(
        res.converged,
        "discrete Poisson failed to converge on the ill-scaled design"
    );

    // And it must converge to the certified maximum-likelihood estimate.
    for (i, &want) in EXPECTED_PARAMS.iter().enumerate() {
        assert_abs_diff_eq!(res.params[i], want, epsilon = 1e-6 * (1.0 + want.abs()));
    }
}
