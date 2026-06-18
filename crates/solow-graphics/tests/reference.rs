//! Cross-validation of the graphics layer's *computed* data against golden
//! reference values frozen in `tests/fixtures/graphics.json`.
//!
//! The SVG output itself is only checked structurally (starts with `<svg`,
//! closes, and contains drawable elements) — never pixel-exact.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_graphics::{acf, conf_band, pacf_yw, plot_acf, plot_pacf, plot_resid_fitted, ProbPlot};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/graphics.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_graphics.py)");
    serde_json::from_str(&s).unwrap()
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

fn check_scalar(label: &str, got: f64, want: f64, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

// --- ProbPlot / qqplot -----------------------------------------------------

#[test]
fn probplot_matches_reference() {
    let fx = load();
    for c in fx["probplots"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let a = c["a"].as_f64().unwrap();
        let data: Vec<f64> = c["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let pp = ProbPlot::with_a(&data, a);

        // Closed-form quantities: tight 1e-8 (ppf goes through a distribution
        // inverse but we still match comfortably below 1e-8).
        check_vec(
            name,
            &pp.theoretical_percentiles(),
            c,
            "theoretical_percentiles",
            1e-10,
        );
        check_vec(
            name,
            &pp.theoretical_quantiles(),
            c,
            "theoretical_quantiles",
            1e-8,
        );
        check_vec(name, &pp.sample_quantiles(), c, "sample_quantiles", 1e-12);

        // qqline fits.
        let r = pp.qqline_regression();
        check_scalar(
            &format!("{name}.qqline_r.slope"),
            r.slope,
            c["qqline_r"]["slope"].as_f64().unwrap(),
            1e-8,
        );
        check_scalar(
            &format!("{name}.qqline_r.intercept"),
            r.intercept,
            c["qqline_r"]["intercept"].as_f64().unwrap(),
            1e-8,
        );

        let s = pp.qqline_standardized();
        check_scalar(
            &format!("{name}.qqline_s.slope"),
            s.slope,
            c["qqline_s"]["slope"].as_f64().unwrap(),
            1e-8,
        );
        check_scalar(
            &format!("{name}.qqline_s.intercept"),
            s.intercept,
            c["qqline_s"]["intercept"].as_f64().unwrap(),
            1e-8,
        );

        let q = pp.qqline_quartile();
        check_scalar(
            &format!("{name}.qqline_q.slope"),
            q.slope,
            c["qqline_q"]["slope"].as_f64().unwrap(),
            1e-8,
        );
        check_scalar(
            &format!("{name}.qqline_q.intercept"),
            q.intercept,
            c["qqline_q"]["intercept"].as_f64().unwrap(),
            1e-8,
        );

        // SVG is only checked structurally.
        let svg = pp.qqplot().to_svg();
        assert!(svg.starts_with("<svg"), "{name}: svg prefix");
        assert!(svg.contains("</svg>"), "{name}: svg close");
        assert!(svg.contains("circle"), "{name}: svg has scatter markers");
    }
}

// --- ACF / PACF ------------------------------------------------------------

#[test]
fn acf_pacf_matches_reference() {
    let fx = load();
    for c in fx["acf"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let nlags = c["nlags"].as_u64().unwrap() as usize;
        let alpha = c["alpha"].as_f64().unwrap();
        let x: Vec<f64> = c["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        // acf and pacf to 1e-8.
        check_vec(name, &acf(&x, nlags), c, "acf", 1e-8);
        check_vec(name, &pacf_yw(&x, nlags), c, "pacf", 1e-8);

        // Confidence band to 1e-8.
        check_scalar(
            &format!("{name}.conf_band"),
            conf_band(x.len(), alpha),
            c["conf_band"].as_f64().unwrap(),
            1e-8,
        );

        // Verify the plotting wrappers return identical arrays + a structural SVG.
        let (fig_a, res_a) = plot_acf(&x, nlags, alpha);
        check_vec(name, &res_a.values, c, "acf", 1e-8);
        check_scalar(
            &format!("{name}.acf.band"),
            res_a.conf_band,
            c["conf_band"].as_f64().unwrap(),
            1e-8,
        );
        let svg_a = fig_a.to_svg();
        assert!(
            svg_a.starts_with("<svg") && svg_a.contains("</svg>"),
            "{name}: acf svg"
        );

        let (fig_p, res_p) = plot_pacf(&x, nlags, alpha);
        check_vec(name, &res_p.values, c, "pacf", 1e-8);
        let svg_p = fig_p.to_svg();
        assert!(
            svg_p.starts_with("<svg") && svg_p.contains("</svg>"),
            "{name}: pacf svg"
        );
    }
}

// --- residuals-vs-fitted ---------------------------------------------------

/// OLS via the normal equations `(X'X) beta = X'y` (the design cases are
/// well-conditioned, so this reproduces the reference fit to machine epsilon).
fn ols_fit(x: &Array2<f64>, y: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);
    let beta = solve_spd(&xtx, &xty);
    let fitted = x.dot(&beta);
    let resid = y - &fitted;
    (fitted, resid)
}

/// Solve a symmetric positive-definite system by Gaussian elimination with
/// partial pivoting (small, well-conditioned systems only).
fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = a.nrows();
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| a[[i, j]]).collect())
        .collect();
    let mut r: Vec<f64> = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        for row in (col + 1)..n {
            if m[row][col].abs() > m[piv][col].abs() {
                piv = row;
            }
        }
        m.swap(col, piv);
        r.swap(col, piv);
        let pivot_row = m[col].clone();
        let d = pivot_row[col];
        let pivot_rhs = r[col];
        for row in (col + 1)..n {
            let f = m[row][col] / d;
            for (mc, &pc) in m[row].iter_mut().zip(pivot_row.iter()).skip(col) {
                *mc -= f * pc;
            }
            r[row] -= f * pivot_rhs;
        }
    }
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut s = r[row];
        for c in (row + 1)..n {
            s -= m[row][c] * x[c];
        }
        x[row] = s / m[row][row];
    }
    Array1::from_vec(x)
}

#[test]
fn resid_fitted_matches_reference() {
    let fx = load();
    for c in fx["resid"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = mat(&c["exog"]);
        let y = vec1(&c["endog"]);
        let (fitted, resid) = ols_fit(&x, &y);

        check_vec(name, &fitted, c, "fittedvalues", 1e-8);
        check_vec(name, &resid, c, "resid", 1e-8);

        let svg = plot_resid_fitted(fitted.as_slice().unwrap(), resid.as_slice().unwrap()).to_svg();
        assert!(svg.starts_with("<svg"), "{name}: resid svg prefix");
        assert!(svg.contains("</svg>"), "{name}: resid svg close");
        assert!(svg.contains("circle"), "{name}: resid svg scatter");
    }
}
