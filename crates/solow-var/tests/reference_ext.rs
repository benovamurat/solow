//! Cross-validation of the VECM, Johansen, and recursive-SVAR estimators
//! against golden reference values frozen in `tests/fixtures/var_ext.json`.

use ndarray::{Array2, Axis};
use serde_json::Value;
use solow_var::{coint_johansen, Deterministic, Svar, Vecm};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/var_ext.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_var_ext.py)");
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

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

fn check_mat(label: &str, got: &Array2<f64>, exp: &Value, key: &str, tol: f64) {
    let want = mat(&exp[key]);
    assert_eq!(
        got.dim(),
        want.dim(),
        "{label}.{key}: shape {:?} vs {:?}",
        got.dim(),
        want.dim()
    );
    for i in 0..got.dim().0 {
        for j in 0..got.dim().1 {
            let e = rel(got[[i, j]], want[[i, j]]);
            assert!(
                e <= tol,
                "{label}.{key}[{i}][{j}]: rel-err {e:.3e} (got {}, want {})",
                got[[i, j]],
                want[[i, j]]
            );
        }
    }
}

fn check_vec_field(label: &str, got: &[f64], want: &Value, tol: f64) {
    let want: Vec<f64> = want
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    assert_eq!(got.len(), want.len(), "{label}: length");
    for i in 0..got.len() {
        let e = rel(got[i], want[i]);
        assert!(
            e <= tol,
            "{label}[{i}]: rel-err {e:.3e} (got {}, want {})",
            got[i],
            want[i]
        );
    }
}

fn det_for(name: &str) -> Deterministic {
    match name {
        "n" => Deterministic::None,
        "co" => Deterministic::ConstantOutside,
        "ci" => Deterministic::ConstantInside,
        other => panic!("unsupported deterministic {other}"),
    }
}

#[test]
fn johansen_matches_reference() {
    let fx = load();
    let data = mat(&fx["data"]);
    for c in fx["johansen"].as_array().unwrap() {
        let det_order = c["det_order"].as_i64().unwrap() as i32;
        let k_ar_diff = c["k_ar_diff"].as_u64().unwrap() as usize;
        let label = format!("johansen(det_order={det_order}, k_ar_diff={k_ar_diff})");
        let res = coint_johansen(&data, det_order, k_ar_diff).unwrap();

        // Eigenvalues and statistics: closed form. Spec requires <= 1e-7; the
        // implementation achieves ~3e-9, asserted here at 1e-8.
        check_vec_field(
            &format!("{label}.eig"),
            res.eig.as_slice().unwrap(),
            &c["eig"],
            1e-8,
        );
        check_vec_field(
            &format!("{label}.lr1"),
            res.lr1.as_slice().unwrap(),
            &c["lr1"],
            1e-8,
        );
        check_vec_field(
            &format!("{label}.lr2"),
            res.lr2.as_slice().unwrap(),
            &c["lr2"],
            1e-8,
        );

        // Critical-value tables: exact lookups.
        check_mat(&label, &res.cvt, c, "cvt", 1e-12);
        check_mat(&label, &res.cvm, c, "cvm", 1e-12);
    }
}

#[test]
fn vecm_matches_reference() {
    let fx = load();
    let data = mat(&fx["data"]);
    for c in fx["vecm"].as_array().unwrap() {
        let k_ar_diff = c["k_ar_diff"].as_u64().unwrap() as usize;
        let coint_rank = c["coint_rank"].as_u64().unwrap() as usize;
        let det = det_for(c["deterministic"].as_str().unwrap());
        let label = format!(
            "vecm(det={}, rank={coint_rank}, k_ar_diff={k_ar_diff})",
            c["deterministic"].as_str().unwrap()
        );
        let res = Vecm::with_deterministic(data.clone(), k_ar_diff, coint_rank, det)
            .unwrap()
            .fit()
            .unwrap();

        // Reduced-rank ML estimates (beta is reference-normalized). Spec
        // requires <= 1e-6; the implementation achieves ~1e-11, asserted at 1e-9.
        check_mat(&label, &res.beta, c, "beta", 1e-9);
        check_mat(&label, &res.alpha, c, "alpha", 1e-9);
        check_mat(&label, &res.gamma, c, "gamma", 1e-9);
        check_mat(&label, &res.sigma_u, c, "sigma_u", 1e-9);

        // Deterministic coefficients, if present.
        let dc = c["det_coef"].as_array().unwrap();
        if !dc.is_empty() {
            check_mat(&label, &res.det_coef, c, "det_coef", 1e-9);
        } else {
            assert_eq!(res.det_coef.dim().1, 0, "{label}: unexpected det_coef");
        }
        let dcc = c["det_coef_coint"].as_array().unwrap();
        if !dcc.is_empty() {
            check_mat(&label, &res.det_coef_coint, c, "det_coef_coint", 1e-9);
        } else {
            assert_eq!(
                res.det_coef_coint.dim().0,
                0,
                "{label}: unexpected det_coef_coint"
            );
        }

        // Log-likelihood: MLE quantity. Spec <= 1e-6; achieved ~2e-13,
        // asserted at 1e-10.
        let llf_want = c["llf"].as_f64().unwrap();
        assert!(
            rel(res.llf, llf_want) <= 1e-10,
            "{label}.llf: rel-err {:.3e} (got {}, want {})",
            rel(res.llf, llf_want),
            res.llf,
            llf_want
        );
    }
}

#[test]
fn svar_matches_reference() {
    let fx = load();
    let data = mat(&fx["data"]);
    for c in fx["svar"].as_array().unwrap() {
        let lags = c["lags"].as_u64().unwrap() as usize;
        let label = format!("svar(lags={lags})");
        let res = Svar::new(data.clone()).unwrap().fit(lags).unwrap();

        // Closed-form recursive impact matrix and residual covariance.
        // Achieved ~4e-14, asserted at 1e-10.
        check_mat(&label, &res.sigma_u_mle, c, "sigma_u_mle", 1e-10);
        check_mat(&label, &res.b, c, "B", 1e-10);

        // B is lower triangular and B Bᵀ reproduces Σ_u.
        let recon = res.b.dot(&res.b.t());
        check_mat(&label, &recon, c, "sigma_u_mle", 1e-10);
        for i in 0..res.neqs {
            for j in (i + 1)..res.neqs {
                assert!(
                    res.b[[i, j]].abs() < 1e-12,
                    "{label}: B not lower triangular"
                );
            }
        }
        // A is the identity.
        let asum: f64 = res.a.sum_axis(Axis(0)).sum();
        assert!((asum - res.neqs as f64).abs() < 1e-12);
    }
}
