//! The central correctness contract of `solow-fit`: a model fit through the
//! formula path must be *bit-for-bit* (≤ 1e-12) the same as the same design
//! built by hand and fed to the estimator directly.
//!
//! For each case the "manual" design is assembled independently of the formula
//! engine — raw numeric columns, a hand-coded treatment-contrast dummy for a
//! categorical, an elementwise interaction product — and the constant is added
//! with [`add_constant`]. We then compare coefficients and standard errors
//! against the formula-driven [`ols`] fit.

use ndarray::{Array1, Array2};
use solow_core::tools::{add_constant, HasConstant};
use solow_fit::{gls, ols, wls, DataFrame};
use solow_regression::LinearModel;

const TOL: f64 = 1e-12;

/// Build a `LinearModel::ols` fit directly from raw column-major data, adding
/// the constant as the first column (matching the formula's `Intercept`).
fn manual_ols(y: Vec<f64>, cols: Vec<Vec<f64>>) -> solow_regression::LinearResults {
    let n = y.len();
    let k = cols.len();
    let mut x = Array2::<f64>::zeros((n, k));
    for (j, c) in cols.iter().enumerate() {
        for i in 0..n {
            x[[i, j]] = c[i];
        }
    }
    let design = add_constant(&x, true, HasConstant::Add).unwrap();
    let endog = Array1::from(y);
    LinearModel::ols(endog, design).unwrap().fit().unwrap()
}

/// Largest absolute element-wise difference of two vectors.
fn max_abs_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_same(
    formula: &solow_regression::LinearResults,
    manual: &solow_regression::LinearResults,
) {
    let dp = max_abs_diff(&formula.params, &manual.params);
    let db = max_abs_diff(&formula.bse, &manual.bse);
    assert!(dp <= TOL, "params differ by {dp} (> {TOL})");
    assert!(db <= TOL, "bse differ by {db} (> {TOL})");
}

#[test]
fn single_numeric_regressor() {
    let y = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());

    let fit = ols("y ~ x", &df).unwrap();
    assert_eq!(fit.names, vec!["Intercept".to_string(), "x".to_string()]);

    let manual = manual_ols(y, vec![x]);
    assert_same(&fit.results, &manual);
}

#[test]
fn two_numeric_regressors() {
    let y = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let x2 = vec![0.5, -1.0, 2.0, 0.0, 3.5, 1.0, -2.0];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x1", x1.clone());
    df.add_numeric("x2", x2.clone());

    let fit = ols("y ~ x1 + x2", &df).unwrap();
    assert_eq!(
        fit.names,
        vec!["Intercept".to_string(), "x1".to_string(), "x2".to_string()]
    );

    let manual = manual_ols(y, vec![x1, x2]);
    assert_same(&fit.results, &manual);
}

#[test]
fn categorical_treatment_contrast() {
    // Three groups a, b, c. Sorted levels => "a" is the dropped reference; the
    // formula emits dummies C(g)[T.b] and C(g)[T.c]. We hand-build exactly
    // those two indicator columns.
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let g = vec!["a", "b", "c", "a", "b", "c", "a", "b", "c"];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_categorical("g", g.clone());

    let fit = ols("y ~ C(g)", &df).unwrap();
    assert_eq!(
        fit.names,
        vec![
            "Intercept".to_string(),
            "C(g)[T.b]".to_string(),
            "C(g)[T.c]".to_string()
        ]
    );

    let dummy_b: Vec<f64> = g
        .iter()
        .map(|s| if *s == "b" { 1.0 } else { 0.0 })
        .collect();
    let dummy_c: Vec<f64> = g
        .iter()
        .map(|s| if *s == "c" { 1.0 } else { 0.0 })
        .collect();
    let manual = manual_ols(y, vec![dummy_b, dummy_c]);
    assert_same(&fit.results, &manual);
}

#[test]
fn categorical_plus_numeric() {
    let y = vec![1.5, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5];
    let x = vec![0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8];
    let g = vec!["lo", "hi", "lo", "hi", "lo", "hi", "lo", "hi"];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());
    df.add_categorical("g", g.clone());

    let fit = ols("y ~ x + C(g)", &df).unwrap();
    // Sorted levels: "hi" < "lo" => "hi" is the reference, dummy is C(g)[T.lo].
    assert_eq!(
        fit.names,
        vec![
            "Intercept".to_string(),
            "C(g)[T.lo]".to_string(),
            "x".to_string()
        ]
    );

    let dummy_lo: Vec<f64> = g
        .iter()
        .map(|s| if *s == "lo" { 1.0 } else { 0.0 })
        .collect();
    // Manual design column order must match the formula's: dummy first, then x.
    let manual = manual_ols(y, vec![dummy_lo, x]);
    assert_same(&fit.results, &manual);
}

#[test]
fn numeric_interaction() {
    let y = vec![1.0, 2.2, 2.9, 4.1, 5.0, 6.2, 6.8, 8.1];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![2.0, 0.5, 1.5, 3.0, 2.5, 1.0, 4.0, 0.0];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x1", x1.clone());
    df.add_numeric("x2", x2.clone());

    // Full cross: x1 + x2 + x1:x2.
    let fit = ols("y ~ x1 * x2", &df).unwrap();
    assert_eq!(
        fit.names,
        vec![
            "Intercept".to_string(),
            "x1".to_string(),
            "x2".to_string(),
            "x1:x2".to_string()
        ]
    );

    let inter: Vec<f64> = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).collect();
    let manual = manual_ols(y, vec![x1, x2, inter]);
    assert_same(&fit.results, &manual);
}

#[test]
fn categorical_numeric_interaction() {
    // Interaction of a numeric with a categorical: x:C(g) expands to one column
    // per non-reference level, each the product of x and that level's dummy.
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 0.5, 1.0, 2.0, 4.0];
    let g = vec!["p", "q", "p", "q", "p", "q", "p", "q", "p", "q"];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());
    df.add_categorical("g", g.clone());

    let fit = ols("y ~ x + x:C(g)", &df).unwrap();

    // x:C(g) with a lower-order x present and an intercept uses the reduced
    // contrast: one interaction column for the non-reference level "q".
    let dummy_q: Vec<f64> = g
        .iter()
        .map(|s| if *s == "q" { 1.0 } else { 0.0 })
        .collect();
    let x_times_q: Vec<f64> = x.iter().zip(dummy_q.iter()).map(|(a, b)| a * b).collect();
    let manual = manual_ols(y.clone(), vec![x.clone(), x_times_q]);

    assert_same(&fit.results, &manual);
    // Sanity on the labels emitted for the interaction.
    assert_eq!(fit.names[0], "Intercept");
    assert!(fit.names.iter().any(|n| n == "x"));
    assert!(fit.names.iter().any(|n| n.contains("x:C(g)")));
}

#[test]
fn wls_matches_manual() {
    let y = vec![1.0, 2.5, 2.0, 4.0, 5.5, 5.0, 7.0];
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = vec![1.0, 2.0, 0.5, 1.5, 3.0, 0.8, 1.2];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());

    let fit = wls("y ~ x", &df, Array1::from(w.clone())).unwrap();

    let mut xm = Array2::<f64>::zeros((x.len(), 1));
    for (i, v) in x.iter().enumerate() {
        xm[[i, 0]] = *v;
    }
    let design = add_constant(&xm, true, HasConstant::Add).unwrap();
    let manual = LinearModel::wls(Array1::from(y), design, Array1::from(w))
        .unwrap()
        .fit()
        .unwrap();
    assert_same(&fit.results, &manual);
}

#[test]
fn gls_matches_manual() {
    let y = vec![1.1, 1.9, 3.2, 3.9, 5.1, 6.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let n = y.len();
    // A simple SPD covariance: AR(1)-like with rho = 0.4.
    let rho: f64 = 0.4;
    let mut sigma = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sigma[[i, j]] = rho.powi((i as i32 - j as i32).abs());
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());

    let fit = gls("y ~ x", &df, &sigma).unwrap();

    let mut xm = Array2::<f64>::zeros((n, 1));
    for (i, v) in x.iter().enumerate() {
        xm[[i, 0]] = *v;
    }
    let design = add_constant(&xm, true, HasConstant::Add).unwrap();
    let manual = LinearModel::gls(Array1::from(y), design, &sigma)
        .unwrap()
        .fit()
        .unwrap();
    assert_same(&fit.results, &manual);
}

#[test]
fn no_response_is_an_error() {
    let mut df = DataFrame::new();
    df.add_numeric("x", vec![1.0, 2.0, 3.0]);
    assert!(ols("~ x", &df).is_err());
}

#[test]
fn report_max_diff_across_formulas() {
    // Aggregate the worst params/bse discrepancy across every formula style and
    // print it, so the actual precision is observable in the test log.
    let y = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let x2 = vec![0.5, -1.0, 2.0, 0.0, 3.5, 1.0, -2.0, 0.7, 1.3];
    let g = vec!["a", "b", "c", "a", "b", "c", "a", "b", "c"];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x1", x1.clone());
    df.add_numeric("x2", x2.clone());
    df.add_categorical("g", g.clone());

    let dummy_b: Vec<f64> = g
        .iter()
        .map(|s| if *s == "b" { 1.0 } else { 0.0 })
        .collect();
    let dummy_c: Vec<f64> = g
        .iter()
        .map(|s| if *s == "c" { 1.0 } else { 0.0 })
        .collect();
    let inter: Vec<f64> = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).collect();

    // (formula, manual columns in formula order)
    let cases: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("y ~ x1", vec![x1.clone()]),
        ("y ~ x1 + x2", vec![x1.clone(), x2.clone()]),
        // C(g) sits in the empty-numeric bucket => before x1.
        (
            "y ~ x1 + C(g)",
            vec![dummy_b.clone(), dummy_c.clone(), x1.clone()],
        ),
        ("y ~ x1 * x2", vec![x1.clone(), x2.clone(), inter.clone()]),
    ];

    let mut worst = 0.0_f64;
    for (formula, cols) in cases {
        let fit = ols(formula, &df).unwrap();
        let manual = manual_ols(y.clone(), cols);
        let d = max_abs_diff(&fit.results.params, &manual.params)
            .max(max_abs_diff(&fit.results.bse, &manual.bse));
        eprintln!("{formula:<18} max |Δ| = {d:.3e}");
        worst = worst.max(d);
    }
    eprintln!("worst across all formulas: {worst:.3e}");
    assert!(worst <= TOL, "worst diff {worst} exceeds {TOL}");
}
