//! Panic-safety tests for the GLM constructors: non-finite endog/exog must
//! yield a clean `Err`, and degenerate designs must not panic.

use ndarray::{array, Array1, Array2};
use solow_core::error::Error;
use solow_glm::{Family, Glm, TweedieGlm};

#[test]
fn glm_rejects_nan_endog() {
    let y = array![0.0, 1.0, f64::NAN, 1.0];
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    assert!(matches!(
        Glm::new(y, x, Family::Binomial),
        Err(Error::Value(_))
    ));
}

#[test]
fn glm_rejects_inf_exog() {
    let y = array![1.0, 2.0, 3.0, 4.0];
    let x = array![[1.0, 0.0], [1.0, f64::INFINITY], [1.0, 2.0], [1.0, 3.0]];
    assert!(matches!(
        Glm::new(y, x, Family::Poisson),
        Err(Error::Value(_))
    ));
}

#[test]
fn tweedie_rejects_nan_endog() {
    let y = array![1.0, f64::NAN, 3.0];
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
    assert!(matches!(TweedieGlm::new(y, x, 1.5), Err(Error::Value(_))));
}

#[test]
fn glm_collinear_design_does_not_panic() {
    // Rank-deficient design (third column == 2 * second).
    let y = array![0.0, 1.0, 0.0, 1.0, 1.0];
    let x = array![
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 4.0],
        [1.0, 3.0, 6.0],
        [1.0, 4.0, 8.0],
        [1.0, 5.0, 10.0],
    ];
    // Must return a result (possibly Err) without panicking.
    let _ = Glm::new(y, x, Family::Binomial).unwrap().fit();
}

#[test]
fn glm_empty_endog_does_not_panic() {
    let y: Array1<f64> = Array1::zeros(0);
    let x: Array2<f64> = Array2::zeros((0, 2));
    // Shape is consistent (0 rows); constructing then fitting must not panic.
    if let Ok(model) = Glm::new(y, x, Family::Gaussian) {
        let _ = model.fit();
    }
}
