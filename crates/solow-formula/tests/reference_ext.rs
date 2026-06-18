//! Cross-validation of the *extended* formula features (extra contrast codings
//! and the `**` / `/` operators) against patsy design matrices frozen in
//! `tests/fixtures/formula_ext.json` (regenerate with
//! `python3 tools/reference/gen_formula_ext.py`).

use ndarray::Array2;
use serde_json::Value;
use solow_formula::{build, DataFrame};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/formula_ext.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_formula_ext.py)");
    serde_json::from_str(&s).unwrap()
}

fn mat(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    let (m, n) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((m, n), rows.into_iter().flatten().collect()).unwrap()
}

fn vec1(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn strvec(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

/// Build a [`DataFrame`] from the fixture's numeric and categorical columns.
fn frame(case: &Value) -> DataFrame {
    let mut df = DataFrame::new();
    if let Some(num) = case["numeric"].as_object() {
        for (name, vals) in num {
            df.add_numeric(name, vec1(vals));
        }
    }
    if let Some(cat) = case["categorical"].as_object() {
        for (name, vals) in cat {
            df.add_categorical(name, strvec(vals));
        }
    }
    df
}

const TOL: f64 = 1e-12;

#[test]
fn formula_ext_matches_patsy() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let formula = c["formula"].as_str().unwrap();
        let df = frame(c);

        let out = build(formula, &df).unwrap_or_else(|e| panic!("{name}: build failed: {e}"));

        // ---- column names must match exactly ----
        let want_names = strvec(&c["names"]);
        assert_eq!(
            out.names, want_names,
            "{name} ({formula}): column names differ\n  got:  {:?}\n  want: {:?}",
            out.names, want_names
        );

        // ---- design values within 1e-12 ----
        let want = mat(&c["design"]);
        assert_eq!(
            out.design.dim(),
            want.dim(),
            "{name}: design shape {:?} != {:?}",
            out.design.dim(),
            want.dim()
        );
        for ((i, j), &g) in out.design.indexed_iter() {
            let w = want[[i, j]];
            assert!(
                (g - w).abs() <= TOL,
                "{name} ({formula}): design[{i}][{j}] = {g}, want {w} (|diff| {:.3e})",
                (g - w).abs()
            );
        }

        // ---- response ----
        match (c["y"].as_array(), &out.y) {
            (Some(_), Some(y)) => {
                let want_y = vec1(&c["y"]);
                assert_eq!(y.len(), want_y.len(), "{name}: response length");
                for (k, (&g, &w)) in y.iter().zip(want_y.iter()).enumerate() {
                    assert!((g - w).abs() <= TOL, "{name}: y[{k}] = {g}, want {w}");
                }
            }
            (None, None) => {}
            (a, b) => panic!(
                "{name}: response presence mismatch (fixture has_y={}, rust has_y={})",
                a.is_some(),
                b.is_some()
            ),
        }
    }
}
