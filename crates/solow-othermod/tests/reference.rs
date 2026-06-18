//! Cross-validation of the beta-regression estimator against golden reference
//! values frozen in `tests/fixtures/othermod.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_othermod::{BetaModel, Link};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/othermod.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_othermod.py)");
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

fn check_scalar(label: &str, got: f64, exp: &Value, key: &str, tol: f64) {
    let want = exp[key].as_f64().unwrap();
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}.{key}: rel-err {e:.3e} (got {got}, want {want})"
    );
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

fn link_for(name: &str) -> Link {
    match name {
        "logit" => Link::Logit,
        "log" => Link::Log,
        other => panic!("unknown link {other}"),
    }
}

#[test]
fn betareg_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let link = link_for(c["link"].as_str().unwrap());
        let link_prec = link_for(c["link_precision"].as_str().unwrap());
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let z = mat(&c["exog_precision"]);
        let res = BetaModel::with_links(y, x, z, link, link_prec)
            .unwrap()
            .fit()
            .unwrap();
        assert!(res.converged, "{name}: did not converge");

        let exp = &c["expected"];
        // MLE point estimates (params): tight.
        check_vec(name, &res.params, exp, "params", 1e-6);
        // Standard errors come from the observed-information inverse.
        check_vec(name, &res.bse, exp, "bse", 1e-6);
        // Derived inference.
        check_vec(name, &res.tvalues, exp, "tvalues", 1e-6);
        check_vec(name, &res.pvalues, exp, "pvalues", 1e-6);
        check_vec(name, &res.fittedvalues, exp, "fittedvalues", 1e-7);
        check_scalar(name, res.llf, exp, "llf", 1e-7);

        // Confidence interval (normal, 95%).
        let ci = mat(&exp["conf_int"]);
        let got = res.conf_int(0.05);
        for i in 0..res.params.len() {
            for j in 0..2 {
                assert!(
                    rel(got[[i, j]], ci[[i, j]]) <= 1e-6,
                    "{name}.conf_int[{i}][{j}]: got {} want {}",
                    got[[i, j]],
                    ci[[i, j]]
                );
            }
        }
    }
}
