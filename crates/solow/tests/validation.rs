//! Real-world validation (M11).
//!
//! This suite proves Solow reproduces *certified and published* results on real
//! datasets, not merely on synthetic fixtures.  It has two independent halves,
//! both driven by `tests/fixtures/validation.json`:
//!
//! 1. **NIST StRD** linear-regression certified benchmarks. The data and the
//!    certified coefficients / standard errors are published by NIST (the U.S.
//!    National Institute of Standards and Technology) to ~15 significant figures
//!    and are transcribed verbatim into the fixture. These checks depend on
//!    NOTHING but NIST — they validate Solow against an external certifying
//!    authority rather than against another software package. Per-dataset
//!    tolerances reflect each benchmark's documented numerical difficulty.
//!
//! 2. **Canonical real example datasets** (Spector & Mazzeo, Longley,
//!    stack-loss, Scotland devolution, capital punishment) fit through the
//!    modeling reference; Solow must reproduce params / bse / llf to ~1e-6.
//!
//! Run with: `cargo test -p solow --test validation`.

use ndarray::{concatenate, Array1, Array2, Axis};
use serde_json::Value;
use solow::discrete::{Logit, Poisson};
use solow::glm::{Family, Glm, Link};
use solow::regression::LinearModel;
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/validation.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_validation.py)");
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

/// Relative error that degrades gracefully to absolute error near zero, so a
/// certified value of exactly 0 (e.g. Wampler1 std-errors) is handled sanely.
fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

/// Prepend a column of ones to a design matrix.
fn prepend_const(x: &Array2<f64>) -> Array2<f64> {
    let ones = Array2::<f64>::ones((x.nrows(), 1));
    concatenate(Axis(1), &[ones.view(), x.view()]).unwrap()
}

// ===========================================================================
// PART 1 — NIST StRD certified benchmarks
// ===========================================================================

/// The worst (largest) relative error a NIST case achieved, reported back so the
/// run prints the *actual* accuracy, not just pass/fail.
struct Achieved {
    name: String,
    max_param_rel: f64,
    max_bse_rel: f64,
    resid_std_rel: f64,
}

fn check_nist(case: &Value, param_tol: f64, bse_tol: f64) -> Achieved {
    let name = case["name"].as_str().unwrap().to_string();
    let has_intercept = case["has_intercept"].as_bool().unwrap();
    let y = vec1(&case["endog"]);
    let raw = mat(&case["exog"]);
    let design = if has_intercept {
        prepend_const(&raw)
    } else {
        raw
    };

    let res = LinearModel::ols(y, design)
        .unwrap_or_else(|e| panic!("{name}: OLS construction failed: {e}"))
        .fit()
        .unwrap_or_else(|e| panic!("{name}: OLS fit failed: {e}"));

    let cert = &case["certified"];
    let want_params = vec1(&cert["params"]);
    let want_bse = vec1(&cert["bse"]);

    assert_eq!(
        res.params.len(),
        want_params.len(),
        "{name}: parameter count mismatch"
    );

    let mut max_param_rel = 0.0_f64;
    for i in 0..want_params.len() {
        let e = rel(res.params[i], want_params[i]);
        max_param_rel = max_param_rel.max(e);
        assert!(
            e <= param_tol,
            "{name}.params[{i}]: rel-err {e:.3e} > tol {param_tol:.0e} \
             (got {}, NIST-certified {})",
            res.params[i],
            want_params[i]
        );
    }

    let mut max_bse_rel = 0.0_f64;
    for i in 0..want_bse.len() {
        let e = rel(res.bse[i], want_bse[i]);
        max_bse_rel = max_bse_rel.max(e);
        assert!(
            e <= bse_tol,
            "{name}.bse[{i}]: rel-err {e:.3e} > tol {bse_tol:.0e} \
             (got {}, NIST-certified {})",
            res.bse[i],
            want_bse[i]
        );
    }

    // Residual standard deviation: NIST certifies sqrt(scale).
    let want_resid_std = cert["residual_std"].as_f64().unwrap();
    let got_resid_std = res.scale.sqrt();
    let resid_std_rel = rel(got_resid_std, want_resid_std);
    assert!(
        resid_std_rel <= bse_tol,
        "{name}.residual_std: rel-err {resid_std_rel:.3e} > tol {bse_tol:.0e} \
         (got {got_resid_std}, NIST-certified {want_resid_std})"
    );

    // R^2 is also certified.
    let want_r2 = cert["rsquared"].as_f64().unwrap();
    let r2_rel = rel(res.rsquared, want_r2);
    assert!(
        r2_rel <= bse_tol,
        "{name}.rsquared: rel-err {r2_rel:.3e} > tol {bse_tol:.0e} \
         (got {}, NIST-certified {want_r2})",
        res.rsquared
    );

    println!(
        "  NIST {name:<10} max|Δparam|={max_param_rel:.2e}  \
         max|Δbse|={max_bse_rel:.2e}  |Δresid_std|={resid_std_rel:.2e}"
    );

    Achieved {
        name,
        max_param_rel,
        max_bse_rel,
        resid_std_rel,
    }
}

#[test]
fn nist_strd_certified() {
    let fx = load();
    let cases: Vec<&Value> = fx["nist"].as_array().unwrap().iter().collect();

    // Per-dataset tolerances reflect the dataset's documented difficulty.
    //
    //   * Norris   — lower difficulty; near machine precision.
    //   * Wampler1 — exact integer polynomial; coefficients reproduced to a few
    //                ulps (params ~1e-9 relative). Its certified std-errors are
    //                all exactly 0, so we use an absolute-style floor there.
    //   * NoInt1   — no-intercept straight line; near machine precision.
    //   * Longley  — the textbook ill-conditioned design (condition number ~1e10);
    //                certified to the published digits at ~1e-6 relative, which is
    //                excellent for a problem this poorly conditioned.
    let mut achieved = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let (ptol, btol) = match name {
            "Norris" => (1e-10, 1e-9),
            "Wampler1" => (1e-7, 1e-6),
            "NoInt1" => (1e-10, 1e-10),
            "Longley" => (1e-6, 1e-6),
            other => panic!("unexpected NIST dataset {other}"),
        };
        achieved.push(check_nist(case, ptol, btol));
    }

    // Sanity: we actually exercised the four required certified datasets.
    let names: Vec<&str> = achieved.iter().map(|a| a.name.as_str()).collect();
    for required in ["Norris", "Longley", "Wampler1", "NoInt1"] {
        assert!(names.contains(&required), "missing NIST dataset {required}");
    }

    // Wampler1 should reproduce the integer coefficients essentially exactly.
    let wampler = achieved.iter().find(|a| a.name == "Wampler1").unwrap();
    assert!(
        wampler.max_param_rel < 1e-7,
        "Wampler1 should reproduce integer coefficients near machine precision, \
         got {:.3e}",
        wampler.max_param_rel
    );

    // Worst-case achieved accuracy across ALL certified NIST quantities — this
    // single number is the headline credibility figure for the linear solver.
    let worst = achieved
        .iter()
        .map(|a| a.max_param_rel.max(a.max_bse_rel).max(a.resid_std_rel))
        .fold(0.0_f64, f64::max);
    println!("  NIST worst-case certified rel-error across all datasets: {worst:.2e}");
    assert!(
        worst <= 1e-6,
        "NIST worst-case rel-error {worst:.3e} exceeds 1e-6"
    );
}

// ===========================================================================
// PART 2 — Canonical real datasets via the reference
// ===========================================================================

fn family_for(name: &str) -> Family {
    match name {
        "Gaussian" => Family::Gaussian,
        "Poisson" => Family::Poisson,
        "Binomial" => Family::Binomial,
        "Gamma" => Family::Gamma,
        other => panic!("unknown family {other}"),
    }
}

fn link_for(name: &str) -> Link {
    match name {
        "identity" => Link::Identity,
        "log" => Link::Log,
        "logit" => Link::Logit,
        "probit" => Link::Probit,
        "inverse_power" => Link::InversePower,
        other => panic!("unknown link {other}"),
    }
}

fn check_reference(case: &Value, params_tol: f64, bse_tol: f64, llf_tol: f64) {
    let name = case["name"].as_str().unwrap();
    let kind = case["kind"].as_str().unwrap();
    let y = vec1(&case["endog"]);
    // The dumped design already carries its constant column (added in the
    // generator), so it is fed straight to the estimator.
    let x = mat(&case["exog"]);
    let exp = &case["expected"];
    let want_params = vec1(&exp["params"]);
    let want_bse = vec1(&exp["bse"]);
    let want_llf = exp["llf"].as_f64().unwrap();

    let (got_params, got_bse, got_llf, converged) = match kind {
        "ols" => {
            let r = LinearModel::ols(y, x).unwrap().fit().unwrap();
            (r.params, r.bse, r.llf, true)
        }
        "logit" => {
            let r = Logit::new(y, x).unwrap().fit().unwrap();
            (r.params, r.bse, r.llf, r.converged)
        }
        "poisson" => {
            let r = Poisson::new(y, x).unwrap().fit().unwrap();
            (r.params, r.bse, r.llf, r.converged)
        }
        "glm" => {
            let family = family_for(case["family"].as_str().unwrap());
            let link = link_for(case["link"].as_str().unwrap());
            let r = Glm::with_link(y, x, family, link).unwrap().fit().unwrap();
            (r.params, r.bse, r.llf, r.converged)
        }
        other => panic!("unknown kind {other}"),
    };

    assert!(converged, "{name}: estimator did not converge");

    let mut max_p = 0.0_f64;
    for i in 0..want_params.len() {
        let e = rel(got_params[i], want_params[i]);
        max_p = max_p.max(e);
        assert!(
            e <= params_tol,
            "{name}.params[{i}]: rel-err {e:.3e} > {params_tol:.0e} \
             (got {}, reference {})",
            got_params[i],
            want_params[i]
        );
    }
    let mut max_b = 0.0_f64;
    for i in 0..want_bse.len() {
        let e = rel(got_bse[i], want_bse[i]);
        max_b = max_b.max(e);
        assert!(
            e <= bse_tol,
            "{name}.bse[{i}]: rel-err {e:.3e} > {bse_tol:.0e} \
             (got {}, reference {})",
            got_bse[i],
            want_bse[i]
        );
    }
    let llf_e = rel(got_llf, want_llf);
    assert!(
        llf_e <= llf_tol,
        "{name}.llf: rel-err {llf_e:.3e} > {llf_tol:.0e} (got {got_llf}, reference {want_llf})"
    );

    println!(
        "  REF  {name:<28} max|Δparams|={max_p:.2e}  max|Δbse|={max_b:.2e}  |Δllf|={llf_e:.2e}"
    );
}

#[test]
fn canonical_real_datasets_match_reference() {
    let fx = load();
    let cases = fx["reference"].as_array().unwrap();
    assert!(
        cases.len() >= 4,
        "expected at least 4 reference datasets, found {}",
        cases.len()
    );
    for case in cases {
        // ~1e-6 relative agreement on real, sometimes messy data. The ill-
        // conditioned Longley design is the loosest at this level; the binary /
        // count / Gamma fits are tighter.
        check_reference(case, 1e-6, 1e-6, 1e-6);
    }
}
