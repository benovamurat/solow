//! Cross-validation of the regression-diagnostic influence statistics and the
//! mosaic-plot geometry against golden reference values frozen in
//! `tests/fixtures/graphics_ext.json` (generated from the authoritative
//! reference `OLSInfluence`).

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_graphics::{mosaic, Influence};
use solow_regression::LinearModel;
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/graphics_ext.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_graphics_ext.py)");
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

fn vec1(v: &Value) -> Array1<f64> {
    Array1::from_vec(
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect(),
    )
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

fn check_vec(label: &str, got: &Array1<f64>, exp: &Value, key: &str, tol: f64) {
    let want = vec1(&exp[key]);
    assert_eq!(got.len(), want.len(), "{label}.{key}: length");
    for i in 0..got.len() {
        let e = rel(got[i], want[i]);
        assert!(
            e <= tol,
            "{label}.{key}[{i}]: rel-err {e:.3e} (got {}, want {})",
            got[i],
            want[i]
        );
    }
}

#[test]
fn influence_matches_reference() {
    let fx = load();
    for c in fx["influence"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = mat(&c["exog"]);
        let y = vec1(&c["endog"]);
        let res = LinearModel::ols(y, x.clone()).unwrap().fit().unwrap();
        let inf = Influence::new(&res, &x);

        // Closed-form quantities verified at tight tolerance (<= 1e-8).
        check_vec(name, &inf.hat_diag, c, "hat_diag", 1e-9);
        check_vec(
            name,
            &inf.resid_studentized_internal,
            c,
            "resid_studentized_internal",
            1e-9,
        );
        check_vec(
            name,
            &inf.resid_studentized_external,
            c,
            "resid_studentized_external",
            1e-9,
        );
        check_vec(name, &inf.cooks_distance, c, "cooks_distance", 1e-9);
        check_vec(name, &inf.dffits, c, "dffits", 1e-9);
    }
}

#[test]
fn mosaic_matches_reference() {
    let fx = load();
    for c in fx["mosaic"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let counts = mat(&c["counts"]);
        let (_fig, m) = mosaic(&counts);

        check_vec(name, &m.row_widths, c, "row_widths", 1e-12);

        let want = mat(&c["cell_heights"]);
        assert_eq!(m.cell_heights.dim(), want.dim(), "{name}.cell_heights dim");
        for i in 0..want.nrows() {
            for j in 0..want.ncols() {
                let e = rel(m.cell_heights[[i, j]], want[[i, j]]);
                assert!(e <= 1e-12, "{name}.cell_heights[{i}][{j}]: rel-err {e:.3e}");
            }
        }
    }
}
