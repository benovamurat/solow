//! Formula-path GLM and discrete fits match the manual design path, and the
//! labeled summaries render with the formula column names.

use ndarray::{Array1, Array2};
use solow_core::tools::{add_constant, HasConstant};
use solow_discrete::{Logit, Poisson, Probit};
use solow_fit::{glm, logit, poisson, probit, DataFrame, Family};

const TOL: f64 = 1e-12;

fn design_with_const(cols: Vec<Vec<f64>>) -> Array2<f64> {
    let n = cols[0].len();
    let k = cols.len();
    let mut x = Array2::<f64>::zeros((n, k));
    for (j, c) in cols.iter().enumerate() {
        for i in 0..n {
            x[[i, j]] = c[i];
        }
    }
    add_constant(&x, true, HasConstant::Add).unwrap()
}

fn max_abs(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn glm_poisson_matches_manual() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 14.0];
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());

    let fit = glm("y ~ x", &df, Family::Poisson, None).unwrap();
    assert_eq!(fit.names, vec!["Intercept".to_string(), "x".to_string()]);

    let design = design_with_const(vec![x]);
    let manual = solow_glm::Glm::new(Array1::from(y), design, Family::Poisson)
        .unwrap()
        .fit()
        .unwrap();

    assert!(max_abs(&fit.results.params, &manual.params) <= TOL);
    assert!(max_abs(&fit.results.bse, &manual.bse) <= TOL);
}

#[test]
fn logit_matches_manual() {
    let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let x = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 0.2];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());

    let fit = logit("y ~ x", &df).unwrap();
    let design = design_with_const(vec![x]);
    let manual = Logit::new(Array1::from(y), design).unwrap().fit().unwrap();

    assert!(max_abs(&fit.results.params, &manual.params) <= TOL);
    assert!(max_abs(&fit.results.bse, &manual.bse) <= TOL);
}

#[test]
fn probit_matches_manual() {
    let y = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let x = vec![-1.0, -0.8, -0.3, 0.1, 0.4, 0.9, -0.2, 1.1, 1.6, 0.7];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());

    let fit = probit("y ~ x", &df).unwrap();
    let design = design_with_const(vec![x]);
    let manual = Probit::new(Array1::from(y), design).unwrap().fit().unwrap();

    assert!(max_abs(&fit.results.params, &manual.params) <= TOL);
    assert!(max_abs(&fit.results.bse, &manual.bse) <= TOL);
}

#[test]
fn poisson_discrete_matches_manual() {
    let y = vec![0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0];
    let x = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];

    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x", x.clone());

    let fit = poisson("y ~ x", &df).unwrap();
    let design = design_with_const(vec![x]);
    let manual = Poisson::new(Array1::from(y), design)
        .unwrap()
        .fit()
        .unwrap();

    assert!(max_abs(&fit.results.params, &manual.params) <= TOL);
    assert!(max_abs(&fit.results.bse, &manual.bse) <= TOL);
}

#[test]
fn summary_uses_formula_names() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let g = vec!["a", "b", "c", "a", "b", "c", "a", "b", "c"];

    let mut df = DataFrame::new();
    df.add_numeric("y", y);
    df.add_categorical("g", g);

    let fit = solow_fit::ols("y ~ C(g)", &df).unwrap();
    let text = fit.summary();
    assert!(
        text.contains("C(g)[T.b]"),
        "summary should label dummies:\n{text}"
    );
    assert!(
        text.contains("C(g)[T.c]"),
        "summary should label dummies:\n{text}"
    );
    assert!(text.contains("Intercept"));
    assert!(text.contains("R-squared"));
}

#[test]
fn glm_summary_renders_z() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0];
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    let mut df = DataFrame::new();
    df.add_numeric("y", y);
    df.add_numeric("x", x);

    let fit = glm("y ~ x", &df, Family::Poisson, None).unwrap();
    let text = fit.summary();
    assert!(text.contains("Intercept"));
    assert!(text.contains("x"));
    // z statistic column header for normal-based inference.
    assert!(text.contains('z'), "expected a z column:\n{text}");
}
