//! Reference-fixture tests for solow-preprocessing.
//!
//! Every scaler here is deterministic (no RNG), so agreement is
//! bit-wise up to `1e-12` — the same bar solow-metrics ships.

use ndarray::{Array2, ArrayView2};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_preprocessing::{
    BinStrategy, KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, NormKind, Normalizer,
    PolynomialFeatures, RobustScaler, StandardScaler,
};

const TOL: f64 = 1e-12;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/preprocessing")
}

fn load(name: &str) -> Value {
    let path = fixtures().join(format!("{name}.json"));
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn f2d(v: &Value) -> Array2<f64> {
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
    let (n_rows, n_cols) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((n_rows, n_cols), rows.into_iter().flatten().collect()).unwrap()
}

fn assert_close_matrix(name: &str, got: ArrayView2<'_, f64>, want: ArrayView2<'_, f64>) {
    assert_eq!(got.dim(), want.dim(), "{name}: shape mismatch");
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            let g = got[[i, j]];
            let w = want[[i, j]];
            let d = (g - w).abs();
            assert!(
                d <= TOL,
                "{name}[{i},{j}]: Rust = {g:.17}, the reference = {w:.17}, diff = {d:.3e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// StandardScaler — the reference uses the population (biased) variance by default;
// we call `population()` on solow's StandardScaler to match.
// ---------------------------------------------------------------------------

#[test]
fn standard_scaler_matches_reference_bitwise() {
    let f = load("standard_scaler");
    let x = f2d(&f["inputs"]["x"]);
    let sc = StandardScaler::fit_with(x.view(), true, true, true).unwrap();
    let expected_mean = f2d(&Value::Array(vec![f["expected"]["mean"].clone()]));
    let expected_scale = f2d(&Value::Array(vec![f["expected"]["scale"].clone()]));
    let got_mean = sc.mean.clone().insert_axis(ndarray::Axis(0));
    let got_scale = sc.scale.clone().insert_axis(ndarray::Axis(0));
    assert_close_matrix("mean", got_mean.view(), expected_mean.view());
    assert_close_matrix("scale", got_scale.view(), expected_scale.view());
    let tr = sc.transform(x.view()).unwrap();
    let exp_tr = f2d(&f["expected"]["transform"]);
    assert_close_matrix("transform", tr.view(), exp_tr.view());
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------

#[test]
fn minmax_scaler_matches_reference_bitwise() {
    let f = load("minmax_scaler");
    let x = f2d(&f["inputs"]["x"]);
    let (sc, tr) = MinMaxScaler::fit_transform(x.view()).unwrap();
    let expected_min = f2d(&Value::Array(vec![f["expected"]["data_min"].clone()]));
    let expected_max = f2d(&Value::Array(vec![f["expected"]["data_max"].clone()]));
    let got_min = sc.data_min.clone().insert_axis(ndarray::Axis(0));
    let got_max = sc.data_max.clone().insert_axis(ndarray::Axis(0));
    assert_close_matrix("data_min", got_min.view(), expected_min.view());
    assert_close_matrix("data_max", got_max.view(), expected_max.view());
    let exp_tr = f2d(&f["expected"]["transform"]);
    assert_close_matrix("transform", tr.view(), exp_tr.view());
}

// ---------------------------------------------------------------------------
// RobustScaler
// ---------------------------------------------------------------------------

#[test]
fn robust_scaler_matches_reference_bitwise() {
    let f = load("robust_scaler");
    let x = f2d(&f["inputs"]["x"]);
    let (sc, tr) = RobustScaler::fit_transform(x.view()).unwrap();
    let expected_center = f2d(&Value::Array(vec![f["expected"]["center"].clone()]));
    let expected_scale = f2d(&Value::Array(vec![f["expected"]["scale"].clone()]));
    let got_center = sc.center.clone().insert_axis(ndarray::Axis(0));
    let got_scale = sc.scale.clone().insert_axis(ndarray::Axis(0));
    assert_close_matrix("center", got_center.view(), expected_center.view());
    assert_close_matrix("scale", got_scale.view(), expected_scale.view());
    let exp_tr = f2d(&f["expected"]["transform"]);
    assert_close_matrix("transform", tr.view(), exp_tr.view());
}

// ---------------------------------------------------------------------------
// MaxAbsScaler
// ---------------------------------------------------------------------------

#[test]
fn maxabs_scaler_matches_reference_bitwise() {
    let f = load("maxabs_scaler");
    let x = f2d(&f["inputs"]["x"]);
    let (sc, tr) = MaxAbsScaler::fit_transform(x.view()).unwrap();
    let expected_max = f2d(&Value::Array(vec![f["expected"]["max_abs"].clone()]));
    let got_max = sc.max_abs.clone().insert_axis(ndarray::Axis(0));
    assert_close_matrix("max_abs", got_max.view(), expected_max.view());
    let exp_tr = f2d(&f["expected"]["transform"]);
    assert_close_matrix("transform", tr.view(), exp_tr.view());
}

// ---------------------------------------------------------------------------
// Normalizer L2
// ---------------------------------------------------------------------------

#[test]
fn normalizer_l2_matches_reference_bitwise() {
    let f = load("normalizer_l2");
    let x = f2d(&f["inputs"]["x"]);
    let tr = Normalizer::new(NormKind::L2).transform(x.view()).unwrap();
    let exp_tr = f2d(&f["expected"]["transform"]);
    assert_close_matrix("transform", tr.view(), exp_tr.view());
}

// ---------------------------------------------------------------------------
// PolynomialFeatures degree=2, include_bias=true
// ---------------------------------------------------------------------------

#[test]
fn polynomial_features_deg2_matches_reference_bitwise() {
    let f = load("polynomial_features_deg2");
    let x = f2d(&f["inputs"]["x"]);
    let poly = PolynomialFeatures::new(2).include_bias(true);
    let tr = poly.fit_transform(x.view()).unwrap();
    let exp_tr = f2d(&f["expected"]["transform"]);
    assert_close_matrix("transform", tr.view(), exp_tr.view());
}

// ---------------------------------------------------------------------------
// KBinsDiscretizer (uniform)
// ---------------------------------------------------------------------------

#[test]
fn kbins_uniform_matches_reference_bins() {
    let f = load("kbins_uniform");
    let x = f2d(&f["inputs"]["x"]);
    let (kb, tr) = KBinsDiscretizer::fit_transform(x.view(), 4, BinStrategy::Uniform, 0).unwrap();
    // Compare bin edges to 1e-12.
    let expected_edges: Vec<Vec<f64>> = f["expected"]["bin_edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(kb.edges.len(), expected_edges.len(), "n_cols mismatch");
    for j in 0..kb.edges.len() {
        assert_eq!(
            kb.edges[j].len(),
            expected_edges[j].len(),
            "column {j} edges length mismatch"
        );
        for (i, (a, b)) in kb.edges[j].iter().zip(expected_edges[j].iter()).enumerate() {
            let d = (a - b).abs();
            assert!(d <= 1e-12, "edges[{j}][{i}]: {a} vs {b} (diff {d})");
        }
    }
    // Bin assignments must match exactly.
    let expected_bins: Vec<Vec<i64>> = f["expected"]["transform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect()
        })
        .collect();
    for i in 0..tr.nrows() {
        for j in 0..tr.ncols() {
            assert_eq!(tr[[i, j]] as i64, expected_bins[i][j]);
        }
    }
}
