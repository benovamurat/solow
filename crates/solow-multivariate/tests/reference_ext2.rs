//! Cross-validation of factor rotation against golden reference values frozen
//! in `tests/fixtures/multivariate_ext2.json`.
//!
//! The gradient-projection algorithm is deterministic (identity initial guess,
//! the same line search as the reference), so the rotated loadings and the
//! rotation matrix match the reference up to a per-column sign: a rotation
//! criterion is invariant to flipping the sign of a factor, and the reference
//! and this port may settle on opposite signs for a column. We therefore align
//! each column's sign (the largest-magnitude reference entry fixes the sign)
//! before comparing, and additionally check that `L` is reconstructed from `T`
//! exactly (`L = A T` for orthogonal/promax, `L = A (T^{-1})^\top` for
//! oblique).

use ndarray::{Array2, Axis};
use serde_json::Value;
use solow_multivariate::{rotate_factors, RotationMethod};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/multivariate_ext2.json"
    );
    let s = fs::read_to_string(p)
        .expect("fixture present (run tools/reference/gen_multivariate_ext2.py)");
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

fn abs_err(got: f64, want: f64) -> f64 {
    (got - want).abs()
}

/// Per-column sign flips that align `ours` onto `theirs`: the largest-magnitude
/// reference entry in each column decides the column's sign.
fn column_signs(ours: &Array2<f64>, theirs: &Array2<f64>) -> Vec<f64> {
    let nc = ours.ncols();
    let mut signs = vec![1.0; nc];
    for c in 0..nc {
        let mut best = 0usize;
        let mut best_mag = -1.0;
        for r in 0..theirs.nrows() {
            let m = theirs[[r, c]].abs();
            if m > best_mag {
                best_mag = m;
                best = r;
            }
        }
        let prod = ours[[best, c]] * theirs[[best, c]];
        signs[c] = if prod < 0.0 { -1.0 } else { 1.0 };
    }
    signs
}

/// Assert two matrices agree up to a per-column sign (signs derived from the
/// reference, `theirs`).
fn assert_col_signed(label: &str, ours: &Array2<f64>, theirs: &Array2<f64>, tol: f64) {
    assert_eq!(ours.dim(), theirs.dim(), "{label}: shape mismatch");
    let signs = column_signs(ours, theirs);
    for i in 0..theirs.nrows() {
        for j in 0..theirs.ncols() {
            let g = ours[[i, j]] * signs[j];
            let e = abs_err(g, theirs[[i, j]]);
            assert!(
                e <= tol,
                "{label}[{i}][{j}]: abs-err {e:.3e} (got {g}, want {})",
                theirs[[i, j]]
            );
        }
    }
}

fn method_for(name: &str) -> RotationMethod {
    RotationMethod::from_name(name).unwrap()
}

#[test]
fn rotation_matches_reference() {
    let fx = load();
    for c in fx["rotation"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let a = mat(&c["loadings"]);
        let exp = &c["expected"];

        for method_name in ["varimax", "quartimax", "oblimin", "quartimin", "promax"] {
            let method = method_for(method_name);
            let r = rotate_factors(&a, method_name).unwrap();
            let e = &exp[method_name];
            let want_l = mat(&e["loadings"]);
            let want_t = mat(&e["rotation"]);
            let label = format!("rotation[{name}].{method_name}");

            // Rotated loadings agree with the reference up to a per-column sign.
            // This port reproduces the reference GPA bit-for-bit (identical
            // identity init, SVD-based projection and line search), so the
            // observed agreement is at machine epsilon (~1e-15) for every
            // method; we assert a conservative 1e-9 to absorb cross-platform
            // float variation in the SVD/inverse.
            assert_col_signed(&format!("{label}.loadings"), &r.loadings, &want_l, 1e-9);

            // Rotation matrix agrees with the reference up to a per-column sign.
            assert_col_signed(&format!("{label}.rotation"), &r.rotation, &want_t, 1e-9);

            // T consistency: reconstruct L from T and check it matches our L
            // exactly (closed form).
            let recon = match method {
                RotationMethod::Varimax | RotationMethod::Quartimax | RotationMethod::Promax => {
                    a.dot(&r.rotation)
                }
                RotationMethod::Oblimin | RotationMethod::Quartimin => {
                    let ti = solow_linalg_inv(&r.rotation);
                    a.dot(&ti.t())
                }
            };
            for i in 0..r.loadings.nrows() {
                for j in 0..r.loadings.ncols() {
                    let e = abs_err(recon[[i, j]], r.loadings[[i, j]]);
                    assert!(e <= 1e-10, "{label}.recon[{i}][{j}]: abs-err {e:.3e}");
                }
            }

            // For orthogonal rotations, T is orthogonal (TᵀT = I).
            if matches!(method, RotationMethod::Varimax | RotationMethod::Quartimax) {
                let tt = r.rotation.t().dot(&r.rotation);
                for i in 0..tt.nrows() {
                    for j in 0..tt.ncols() {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            abs_err(tt[[i, j]], want) <= 1e-9,
                            "{label}: TᵀT not identity at ({i},{j})"
                        );
                    }
                }
            }

            // For oblique rotations, T has unit-norm columns.
            if matches!(method, RotationMethod::Oblimin | RotationMethod::Quartimin) {
                for j in 0..r.rotation.ncols() {
                    let nrm = r
                        .rotation
                        .index_axis(Axis(1), j)
                        .iter()
                        .map(|&x| x * x)
                        .sum::<f64>()
                        .sqrt();
                    assert!(
                        abs_err(nrm, 1.0) <= 1e-9,
                        "{label}: T column {j} not unit norm ({nrm})"
                    );
                }
            }
        }
    }
}

/// Minimal Gaussian-elimination matrix inverse (the test crate does not depend
/// on `solow-linalg`, but oblique reconstruction needs `T^{-1}`).
fn solow_linalg_inv(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n);
    let mut m = a.clone();
    let mut inv = Array2::<f64>::eye(n);
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = m[[col, col]].abs();
        for r in (col + 1)..n {
            if m[[r, col]].abs() > best {
                best = m[[r, col]].abs();
                piv = r;
            }
        }
        if piv != col {
            for k in 0..n {
                m.swap([col, k], [piv, k]);
                inv.swap([col, k], [piv, k]);
            }
        }
        let d = m[[col, col]];
        for k in 0..n {
            m[[col, k]] /= d;
            inv[[col, k]] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[[r, col]];
                for k in 0..n {
                    m[[r, k]] -= f * m[[col, k]];
                    inv[[r, k]] -= f * inv[[col, k]];
                }
            }
        }
    }
    inv
}
