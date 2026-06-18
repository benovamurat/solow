//! Focused unit tests for parsing, intercept handling, `I(...)` arithmetic, and
//! error reporting -- complementing the patsy cross-validation in `reference.rs`.

use solow_formula::{build, DataFrame};

fn small() -> DataFrame {
    let mut df = DataFrame::new();
    df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]);
    df.add_numeric("x1", vec![1.0, 2.0, 3.0, 4.0]);
    df.add_numeric("x2", vec![0.0, 1.0, 0.0, 1.0]);
    df.add_categorical("g", vec!["a", "b", "a", "c"]);
    df
}

#[test]
fn intercept_present_by_default() {
    let df = small();
    let out = build("y ~ x1", &df).unwrap();
    assert_eq!(out.names, vec!["Intercept", "x1"]);
    // Intercept column is all ones.
    for v in out.design.column(0) {
        assert_eq!(*v, 1.0);
    }
}

#[test]
fn drop_intercept_minus_one() {
    let df = small();
    let out = build("y ~ x1 - 1", &df).unwrap();
    assert_eq!(out.names, vec!["x1"]);
}

#[test]
fn drop_intercept_plus_zero() {
    let df = small();
    let out = build("y ~ x1 + 0", &df).unwrap();
    assert_eq!(out.names, vec!["x1"]);
}

#[test]
fn no_intercept_full_dummy_coding() {
    let df = small();
    let out = build("y ~ C(g) - 1", &df).unwrap();
    assert_eq!(out.names, vec!["C(g)[a]", "C(g)[b]", "C(g)[c]"]);
}

#[test]
fn star_expands_to_main_effects_and_interaction() {
    let df = small();
    let out = build("y ~ x1*x2", &df).unwrap();
    assert_eq!(out.names, vec!["Intercept", "x1", "x2", "x1:x2"]);
    // x1:x2 column equals product of x1 and x2.
    let x1 = df_col(&out, "x1");
    let x2 = df_col(&out, "x2");
    let inter = df_col(&out, "x1:x2");
    for i in 0..inter.len() {
        assert_eq!(inter[i], x1[i] * x2[i]);
    }
}

#[test]
fn identity_arithmetic_precedence() {
    let df = small();
    // 2*x1 + 1 must apply `*` before `+`.
    let out = build("y ~ I(2*x1 + 1)", &df).unwrap();
    assert_eq!(out.names, vec!["Intercept", "I(2 * x1 + 1)"]);
    let col = df_col(&out, "I(2 * x1 + 1)");
    let x1 = [1.0, 2.0, 3.0, 4.0];
    for i in 0..col.len() {
        assert!((col[i] - (2.0 * x1[i] + 1.0)).abs() < 1e-12);
    }
}

#[test]
fn identity_power() {
    let df = small();
    let out = build("y ~ I(x1**2)", &df).unwrap();
    assert_eq!(out.names, vec!["Intercept", "I(x1 ** 2)"]);
    let col = df_col(&out, "I(x1 ** 2)");
    let x1 = [1.0, 2.0, 3.0, 4.0];
    for i in 0..col.len() {
        assert!((col[i] - x1[i] * x1[i]).abs() < 1e-12);
    }
}

#[test]
fn response_optional() {
    let df = small();
    let out = build("x1 + x2", &df).unwrap();
    assert!(out.y.is_none());
    assert_eq!(out.names, vec!["Intercept", "x1", "x2"]);
}

#[test]
fn response_captured() {
    let df = small();
    let out = build("y ~ x1", &df).unwrap();
    let y = out.y.expect("response present");
    assert_eq!(y.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn minus_removes_term() {
    let df = small();
    let out = build("y ~ x1 + x2 - x2", &df).unwrap();
    assert_eq!(out.names, vec!["Intercept", "x1"]);
}

#[test]
fn unknown_variable_errors() {
    let df = small();
    let err = build("y ~ nope", &df).unwrap_err();
    assert!(format!("{err}").contains("nope"), "got: {err}");
}

#[test]
fn categorical_used_numerically_errors() {
    let df = small();
    let err = build("y ~ g", &df).unwrap_err();
    assert!(format!("{err}").contains("categorical"), "got: {err}");
}

#[test]
fn malformed_formula_errors() {
    let df = small();
    assert!(build("y ~ x1 +", &df).is_err());
    assert!(build("y ~ C(", &df).is_err());
    assert!(build("y ~ I(x1 +)", &df).is_err());
}

// Helper: fetch a named column from the design output as a Vec.
fn df_col(out: &solow_formula::DesignOutput, name: &str) -> Vec<f64> {
    let idx = out
        .names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("no column {name} in {:?}", out.names));
    out.design.column(idx).to_vec()
}
