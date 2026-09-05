//! Classical empirical (sample) covariance.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Sample covariance and mean.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalCovariance {
    /// Fitted mean.
    pub location: Array1<f64>,
    /// Fitted covariance `(d × d)`.
    pub covariance: Array2<f64>,
    /// Sample size at fit time.
    pub n: usize,
    /// Whether the population (biased, `n`) or sample (unbiased, `n − 1`)
    /// divisor was used.
    pub biased: bool,
}

impl EmpiricalCovariance {
    /// Fit with the reference default: population (biased) divisor.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, true)
    }

    /// Full-configuration fit.
    pub fn fit_with(x: ArrayView2<'_, f64>, biased: bool) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "EmpiricalCovariance::fit_with: x must be non-empty".into(),
            ));
        }
        let (n, d) = (x.nrows(), x.ncols());
        let mut mean = Array1::<f64>::zeros(d);
        for j in 0..d {
            for i in 0..n {
                mean[j] += x[[i, j]];
            }
            mean[j] /= n as f64;
        }
        let mut cov = Array2::<f64>::zeros((d, d));
        for i in 0..n {
            for j in 0..d {
                for k in 0..d {
                    cov[[j, k]] += (x[[i, j]] - mean[j]) * (x[[i, k]] - mean[k]);
                }
            }
        }
        let div = if biased {
            n as f64
        } else {
            (n as f64 - 1.0).max(1.0)
        };
        for j in 0..d {
            for k in 0..d {
                cov[[j, k]] /= div;
            }
        }
        Ok(Self {
            location: mean,
            covariance: cov,
            n,
            biased,
        })
    }

    /// Squared Mahalanobis distance of every row of `x` to the fitted
    /// location; returns `NaN` on rows the fit's covariance flags as
    /// numerically singular.
    pub fn mahalanobis(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.location.len() {
            return Err(Error::Shape(format!(
                "EmpiricalCovariance::mahalanobis: expected {} cols, got {}",
                self.location.len(),
                x.ncols()
            )));
        }
        let inv = invert_symmetric(&self.covariance)?;
        let mut out = Array1::<f64>::zeros(x.nrows());
        let d = x.ncols();
        for i in 0..x.nrows() {
            let mut diff = vec![0.0_f64; d];
            for j in 0..d {
                diff[j] = x[[i, j]] - self.location[j];
            }
            let mut s = 0.0_f64;
            for j in 0..d {
                let mut tmp = 0.0_f64;
                for k in 0..d {
                    tmp += inv[[j, k]] * diff[k];
                }
                s += diff[j] * tmp;
            }
            out[i] = s;
        }
        Ok(out)
    }
}

pub(crate) fn invert_symmetric(m: &Array2<f64>) -> Result<Array2<f64>> {
    // Gauss-Jordan on [M | I]; adequate for the small-to-moderate d
    // this crate targets. Adds a small ridge if the pivot vanishes.
    let n = m.nrows();
    let mut a = vec![vec![0.0_f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[[i, j]];
        }
        a[i][n + i] = 1.0;
    }
    for i in 0..n {
        let mut pivot = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                pivot = r;
            }
        }
        if best < 1e-300 {
            return Err(Error::Value(
                "invert_symmetric: matrix is numerically singular".into(),
            ));
        }
        if pivot != i {
            a.swap(i, pivot);
        }
        let piv = a[i][i];
        for c in 0..(2 * n) {
            a[i][c] /= piv;
        }
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = a[r][i];
            if f == 0.0 {
                continue;
            }
            for c in 0..(2 * n) {
                a[r][c] -= f * a[i][c];
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = a[i][n + j];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn empirical_matches_hand_derived_2d() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let e = EmpiricalCovariance::fit(x.view()).unwrap();
        assert!((e.location[0] - 3.0).abs() < 1e-12);
        assert!((e.location[1] - 4.0).abs() < 1e-12);
        // Biased sample cov of this identity-column dataset: var(col) = 8/3.
        assert!((e.covariance[[0, 0]] - 8.0 / 3.0).abs() < 1e-12);
        assert!((e.covariance[[1, 1]] - 8.0 / 3.0).abs() < 1e-12);
        assert!((e.covariance[[0, 1]] - 8.0 / 3.0).abs() < 1e-12);
    }
}
