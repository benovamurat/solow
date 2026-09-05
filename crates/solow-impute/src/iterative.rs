//! IterativeImputer — multivariate imputation via chained regressions
//! (MICE — van Buuren 1999).
//!
//! Each iteration cycles through the columns; for each column with
//! missing values we fit a linear regression against the remaining
//! columns on the complete rows, then substitute the predictions on the
//! incomplete rows. Continues until the maximum absolute change falls
//! below `tol` or `max_iter` sweeps have run.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted IterativeImputer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct IterativeImputer {
    /// Column mean used at the initial-fill step.
    pub initial_fill: Vec<f64>,
    /// Number of sweeps run at fit time.
    pub n_iter: usize,
    /// Whether the fixed point was reached.
    pub converged: bool,
    /// Column count.
    pub n_features_in: usize,
}

impl IterativeImputer {
    /// Fit with the reference defaults `max_iter = 10`, `tol = 1e-3`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>)> {
        Self::fit_with(x, 10, 1e-3)
    }

    /// Full-configuration fit. Returns both the fitted imputer and the
    /// filled matrix.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<(Self, Array2<f64>)> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 || d == 0 {
            return Err(Error::Value("IterativeImputer: empty input".into()));
        }
        // Initial column means.
        let mut init = vec![0.0_f64; d];
        for j in 0..d {
            let observed: Vec<f64> = (0..n).map(|i| x[[i, j]]).filter(|v| v.is_finite()).collect();
            init[j] = if observed.is_empty() {
                0.0
            } else {
                observed.iter().sum::<f64>() / observed.len() as f64
            };
        }
        // Start from column-mean imputation.
        let mut cur = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                cur[[i, j]] = if x[[i, j]].is_finite() {
                    x[[i, j]]
                } else {
                    init[j]
                };
            }
        }
        let mut iters = 0_usize;
        let mut converged = false;
        for it in 0..max_iter {
            iters = it + 1;
            let prev = cur.clone();
            for j in 0..d {
                // Rows with j observed become training; rows with j missing
                // become the imputation targets.
                let train_rows: Vec<usize> = (0..n).filter(|&i| x[[i, j]].is_finite()).collect();
                let target_rows: Vec<usize> = (0..n).filter(|&i| !x[[i, j]].is_finite()).collect();
                if target_rows.is_empty() || train_rows.is_empty() {
                    continue;
                }
                // Build "other-column" design matrices from the current fill.
                let mut xa = Array2::<f64>::zeros((train_rows.len(), d));
                let mut ya = vec![0.0_f64; train_rows.len()];
                for (r, &row) in train_rows.iter().enumerate() {
                    for k in 0..d {
                        if k == j {
                            continue;
                        }
                        xa[[r, k]] = cur[[row, k]];
                    }
                    xa[[r, j]] = 1.0; // intercept column
                    ya[r] = x[[row, j]];
                }
                let Ok(beta) = ols_solve(&xa, &ya) else {
                    continue;
                };
                for &row in &target_rows {
                    let mut pred = beta[j]; // intercept slot
                    for k in 0..d {
                        if k == j {
                            continue;
                        }
                        pred += cur[[row, k]] * beta[k];
                    }
                    cur[[row, j]] = pred;
                }
            }
            let mut delta = 0.0_f64;
            for i in 0..n {
                for j in 0..d {
                    delta += (cur[[i, j]] - prev[[i, j]]).abs();
                }
            }
            if delta < tol * (n * d) as f64 {
                converged = true;
                break;
            }
        }
        Ok((
            Self {
                initial_fill: init,
                n_iter: iters,
                converged,
                n_features_in: d,
            },
            cur,
        ))
    }
}

fn ols_solve(x: &Array2<f64>, y: &[f64]) -> Result<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let mut xtx = vec![vec![0.0_f64; p]; p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0_f64;
            for r in 0..n {
                s += x[[r, i]] * x[[r, j]];
            }
            xtx[i][j] = s;
        }
    }
    // Add a small ridge for stability.
    for i in 0..p {
        xtx[i][i] += 1e-6;
    }
    let mut xty = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = 0.0_f64;
        for r in 0..n {
            s += x[[r, i]] * y[r];
        }
        xty[i] = s;
    }
    // Gauss-Jordan.
    let mut a: Vec<Vec<f64>> = vec![vec![0.0_f64; 2 * p]; p];
    for i in 0..p {
        for j in 0..p {
            a[i][j] = xtx[i][j];
        }
        a[i][p + i] = 1.0;
    }
    for i in 0..p {
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..p {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        if best < 1e-30 {
            return Err(Error::Value("iterative::ols_solve: singular".into()));
        }
        if piv != i {
            a.swap(i, piv);
        }
        let d = a[i][i];
        for c in 0..(2 * p) {
            a[i][c] /= d;
        }
        for r in 0..p {
            if r == i {
                continue;
            }
            let f = a[r][i];
            if f == 0.0 {
                continue;
            }
            for c in 0..(2 * p) {
                a[r][c] -= f * a[i][c];
            }
        }
    }
    let mut sol = vec![0.0_f64; p];
    for i in 0..p {
        for j in 0..p {
            sol[i] += a[i][p + j] * xty[j];
        }
    }
    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn iterative_imputer_fills_missing_values_finitely() {
        let x = array![
            [1.0_f64, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, f64::NAN, 5.0],
            [4.0, 5.0, f64::NAN],
            [5.0, 6.0, 7.0]
        ];
        let (m, filled) = IterativeImputer::fit_with(x.view(), 20, 1e-6).unwrap();
        assert_eq!(m.n_features_in, 3);
        assert!(filled[[2, 1]].is_finite());
        assert!(filled[[3, 2]].is_finite());
    }
}
