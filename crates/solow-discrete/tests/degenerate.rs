//! Panic-safety tests for the discrete-model constructors: non-finite
//! endog/exog must yield a clean `Err`, and degenerate designs must not panic.

use ndarray::{array, Array1, Array2};
use solow_core::error::Error;
use solow_discrete::{Logit, Poisson, Probit};

#[test]
fn logit_rejects_nan_exog() {
    let y = array![0.0, 1.0, 0.0, 1.0];
    let x = array![[1.0, 0.0], [1.0, f64::NAN], [1.0, 2.0], [1.0, 3.0]];
    assert!(matches!(Logit::new(y, x), Err(Error::Value(_))));
}

#[test]
fn probit_rejects_inf_endog() {
    let y = array![0.0, f64::INFINITY, 0.0, 1.0];
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    assert!(matches!(Probit::new(y, x), Err(Error::Value(_))));
}

#[test]
fn poisson_rejects_nan_endog() {
    let y = array![1.0, 2.0, f64::NAN, 4.0];
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    assert!(matches!(Poisson::new(y, x), Err(Error::Value(_))));
}

#[test]
fn logit_single_row_does_not_panic() {
    let y = array![1.0];
    let x = array![[1.0]];
    if let Ok(model) = Logit::new(y, x) {
        let _ = model.fit();
    }
}

#[test]
fn poisson_empty_does_not_panic() {
    let y: Array1<f64> = Array1::zeros(0);
    let x: Array2<f64> = Array2::zeros((0, 1));
    if let Ok(model) = Poisson::new(y, x) {
        let _ = model.fit();
    }
}
