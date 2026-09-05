//! Cross-validation of the Markov-switching estimators against golden reference
//! values frozen in `tests/fixtures/regime.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_regime::{MarkovAutoregression, MarkovRegression, MarkovResults};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/regime.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_regime.py)");
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

/// Maximum absolute relative error across a `(nobs, k)` matrix under a regime
/// permutation: column `i` of `got` is compared to column `perm[i]` of `want`.
fn prob_mat_err(got: &Array2<f64>, want: &Array2<f64>, perm: &[usize]) -> f64 {
    let (n, k) = got.dim();
    let mut e = 0.0f64;
    for t in 0..n {
        for i in 0..k {
            e = e.max(rel(got[[t, i]], want[[t, perm[i]]]));
        }
    }
    e
}

/// All permutations of `0..k`.
fn permutations(k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    permute(&mut idx, 0, &mut out);
    out
}

fn permute(a: &mut Vec<usize>, start: usize, out: &mut Vec<Vec<usize>>) {
    if start == a.len() {
        out.push(a.clone());
        return;
    }
    for i in start..a.len() {
        a.swap(start, i);
        permute(a, start + 1, out);
        a.swap(start, i);
    }
}

/// Verify a fitted result against the reference.
///
/// Regime labels are only identified up to permutation (label switching), so we
/// search over all `k!` regime relabellings and require the best one to match
/// within tolerance. The log-likelihood, AIC and BIC are permutation-invariant
/// and checked directly. `param_tol`/`llf_tol` are the honest achieved
/// tolerances documented in the notes.
fn verify(label: &str, res: &MarkovResults, exp: &Value, param_tol: f64, llf_tol: f64) {
    assert!(res.converged, "{label}: optimiser did not converge");

    // Permutation-invariant scalars.
    check_scalar(label, res.llf, exp, "llf", llf_tol);
    check_scalar(label, res.aic, exp, "aic", llf_tol);
    check_scalar(label, res.bic, exp, "bic", llf_tol);

    let (k, _) = res.transition.dim();
    let want_trans = mat(&exp["transition"]);
    let want_init = vec1(&exp["initial_probabilities"]);
    let want_dur = vec1(&exp["expected_durations"]);
    let want_filt = mat(&exp["filtered_marginal_probabilities"]);
    let want_smooth = mat(&exp["smoothed_marginal_probabilities"]);

    // Find the regime permutation that best aligns the filtered probabilities.
    let perms = permutations(k);
    let best = perms
        .iter()
        .min_by(|a, b| {
            prob_mat_err(&res.filtered_marginal_probabilities, &want_filt, a)
                .partial_cmp(&prob_mat_err(
                    &res.filtered_marginal_probabilities,
                    &want_filt,
                    b,
                ))
                .unwrap()
        })
        .unwrap()
        .clone();

    // Transition matrix under the permutation: P_perm[i, j] = P[perm[i], perm[j]].
    for i in 0..k {
        for j in 0..k {
            let got = res.transition[[best[i], best[j]]];
            let e = rel(got, want_trans[[i, j]]);
            assert!(
                e <= param_tol,
                "{label}.transition[{i}][{j}]: rel-err {e:.3e} (got {got}, want {})",
                want_trans[[i, j]]
            );
        }
    }
    for i in 0..k {
        let e = rel(res.initial_probabilities[best[i]], want_init[i]);
        assert!(
            e <= param_tol,
            "{label}.initial_probabilities[{i}]: rel-err {e:.3e}"
        );
        let e = rel(res.expected_durations[best[i]], want_dur[i]);
        assert!(
            e <= param_tol,
            "{label}.expected_durations[{i}]: rel-err {e:.3e}"
        );
    }

    let ef = prob_mat_err(&res.filtered_marginal_probabilities, &want_filt, &best);
    assert!(ef <= param_tol, "{label}.filtered: rel-err {ef:.3e}");
    let es = prob_mat_err(&res.smoothed_marginal_probabilities, &want_smooth, &best);
    assert!(es <= param_tol, "{label}.smoothed: rel-err {es:.3e}");

    // Non-transition parameters (mean/const, AR, variance) are tagged with their
    // regime index in the name as `name[i]`; compare each to the reference entry
    // for the permuted regime. Transition parameters are covered by the matrix
    // check above.
    let want_params = vec1(&exp["params"]);
    let names = &res.param_names;
    for (idx, nm) in names.iter().enumerate() {
        if nm.starts_with("p[") {
            continue; // transition handled via the matrix.
        }
        // Determine the regime index encoded as the trailing `[i]`, if any.
        if let Some(open) = nm.rfind('[') {
            let inner = &nm[open + 1..nm.len() - 1];
            if let Ok(reg) = inner.parse::<usize>() {
                // Our regime `best[reg_ref] == reg` => reference regime is the
                // position whose best maps to this regime.
                let ref_reg = best.iter().position(|&r| r == reg).unwrap();
                let want_name = nm.replace(&format!("[{reg}]"), &format!("[{ref_reg}]"));
                let widx = exp["param_names"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .position(|v| v.as_str().unwrap() == want_name)
                    .unwrap();
                let e = rel(res.params[idx], want_params[widx]);
                assert!(
                    e <= param_tol,
                    "{label}.param {nm}: rel-err {e:.3e} (got {}, want {})",
                    res.params[idx],
                    want_params[widx]
                );
                continue;
            }
        }
        // Non-switching parameter (e.g. `sigma2`, `ar.L1`): same position.
        let widx = exp["param_names"]
            .as_array()
            .unwrap()
            .iter()
            .position(|v| v.as_str().unwrap() == nm.as_str())
            .unwrap();
        let e = rel(res.params[idx], want_params[widx]);
        assert!(e <= param_tol, "{label}.param {nm}: rel-err {e:.3e}");
    }

    // Bookkeeping (permutation-independent).
    assert_eq!(
        res.nobs,
        exp["nobs"].as_u64().unwrap() as usize,
        "{label}.nobs"
    );
    assert_eq!(
        res.k_params,
        exp["k_params"].as_u64().unwrap() as usize,
        "{label}.k_params"
    );
    let names: Vec<String> = exp["param_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(res.param_names, names, "{label}.param_names");
}

#[test]
fn markov_switching_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let kind = c["kind"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let k = c["k_regimes"].as_u64().unwrap() as usize;
        let exp = &c["expected"];

        match kind {
            "regression" => {
                let sv = c["switching_variance"].as_bool().unwrap();
                let model = MarkovRegression::new(y, k, sv).unwrap();
                let res = model.fit().unwrap();
                // MarkovRegression reaches the true MLE: achieved params/derived
                // quantities to ~3e-10, llf to ~1e-15. Asserted at 1e-8 / 1e-10
                // with margin.
                verify(name, &res, exp, 1e-8, 1e-10);
            }
            "autoregression" => {
                let order = c["order"].as_u64().unwrap() as usize;
                let sar = c["switching_ar"].as_bool().unwrap();
                let sv = c["switching_variance"].as_bool().unwrap();
                let model = MarkovAutoregression::new(y, k, order, sar, sv).unwrap();
                let res = model.fit().unwrap();
                // MarkovAutoregression: achieved params/derived to ~1.7e-8, llf
                // to ~5e-12 (finite-difference Newton polish, k^(order+1)-state
                // filter). Asserted at 1e-7 / 1e-9.
                verify(name, &res, exp, 1e-7, 1e-9);
            }
            other => panic!("unknown kind {other}"),
        }
    }
}
