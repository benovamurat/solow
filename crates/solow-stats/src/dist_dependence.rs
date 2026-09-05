//! Distance covariance and distance correlation (Székely et al., 2007).
//!
//! The distance covariance `dCov(X, Y)` and distance correlation `dCor(X, Y)`
//! measure dependence between two random vectors of arbitrary (possibly
//! different) dimension. Unlike the Pearson correlation they are zero *iff* the
//! variables are independent, and so detect nonlinear and non-monotone
//! dependence. This module mirrors the reference
//! `stats.dist_dependence_measures`.
//!
//! For `n` observations let `a_{ij} = ‖x_i − x_j‖` and `b_{ij} = ‖y_i − y_j‖`
//! be the euclidean-distance matrices, doubly-centered to `A` and `B`:
//!
//! ```text
//! A_{ij} = a_{ij} − ā_{i·} − ā_{·j} + ā_{··}
//! ```
//!
//! Then `dCov = sqrt(mean(A ∘ B))`, the distance variances are
//! `dVar_x = sqrt(mean(A ∘ A))`, `dVar_y = sqrt(mean(B ∘ B))`, and
//! `dCor = dCov / sqrt(dVar_x · dVar_y)`. The basic dCov-test statistic is
//! `n · dCov²`. All quantities are closed-form functions of the distance
//! matrices, so they reproduce the reference to machine precision.

use ndarray::{Array1, Array2};
use solow_core::{Error, Result};
use solow_distributions::norm_cdf;

/// The full battery of distance-dependence statistics for a pair `(X, Y)`.
#[derive(Debug, Clone)]
pub struct DistDependStat {
    /// The basic test statistic `n · dCov²`.
    pub test_statistic: f64,
    /// The distance correlation `dCor(X, Y)` in `[0, 1]`.
    pub distance_correlation: f64,
    /// The distance covariance `dCov(X, Y)`.
    pub distance_covariance: f64,
    /// The distance variance of `X`, `dVar_x`.
    pub dvar_x: f64,
    /// The distance variance of `Y`, `dVar_y`.
    pub dvar_y: f64,
    /// `S = ā_{··} · b̄_{··}`, the product of the grand means of the two
    /// distance matrices (used by the asymptotic test).
    pub s: f64,
}

/// Result of the asymptotic distance-covariance (dCov) test of independence.
#[derive(Debug, Clone)]
pub struct DcovTest {
    /// The asymptotic test statistic `sqrt(n · dCov² / S)`.
    pub statistic: f64,
    /// Two-sided p-value from the standard normal approximation.
    pub pvalue: f64,
}

/// Euclidean pairwise-distance matrix of the rows of `x` (`n × n`).
fn distance_matrix(x: &Array2<f64>) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for c in 0..k {
                let diff = x[[i, c]] - x[[j, c]];
                s += diff * diff;
            }
            let dist = s.sqrt();
            d[[i, j]] = dist;
            d[[j, i]] = dist;
        }
    }
    d
}

/// Double-centering: `A_{ij} = a_{ij} − rowmean_i − colmean_j + grandmean`.
/// Returns `(centered, grand_mean)`.
fn double_center(a: &Array2<f64>) -> (Array2<f64>, f64) {
    let n = a.nrows();
    let nf = n as f64;
    // Column means (axis 0) and row means (axis 1). The matrix is symmetric so
    // row means equal column means, but we follow the reference layout exactly.
    let mut row_means = Array1::<f64>::zeros(n);
    let mut col_means = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut rs = 0.0;
        let mut cs = 0.0;
        for j in 0..n {
            rs += a[[i, j]]; // row i sum (mean over axis 1)
            cs += a[[j, i]]; // column i sum (mean over axis 0)
        }
        row_means[i] = rs / nf;
        col_means[i] = cs / nf;
    }
    let grand: f64 = a.iter().sum::<f64>() / (nf * nf);
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            // Reference: A = a - a_row_means - a_col_means + a_mean, with
            // a_row_means broadcast over rows (axis-0 reduction, length n,
            // indexed by column j) and a_col_means over columns (axis-1
            // reduction, indexed by row i).
            out[[i, j]] = a[[i, j]] - col_means[j] - row_means[i] + grand;
        }
    }
    (out, grand)
}

fn hadamard_mean(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..n {
            s += a[[i, j]] * b[[i, j]];
        }
    }
    s / (n * n) as f64
}

/// Compute every distance-dependence statistic for matched samples `x` and `y`.
///
/// Each of `x` and `y` is an `n × p` matrix whose rows are observations and
/// whose columns are the components of the random vector. The two must share the
/// number of rows (observations) but may differ in the number of columns.
pub fn distance_statistics(x: &Array2<f64>, y: &Array2<f64>) -> Result<DistDependStat> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(Error::Shape(
            "x and y must have the same number of observations (rows)".into(),
        ));
    }
    if n == 0 {
        return Err(Error::Value("empty sample".into()));
    }
    let a = distance_matrix(x);
    let b = distance_matrix(y);
    let (ac, a_mean) = double_center(&a);
    let (bc, b_mean) = double_center(&b);

    let s = a_mean * b_mean;
    let dcov = hadamard_mean(&ac, &bc).sqrt();
    let dvar_x = hadamard_mean(&ac, &ac).sqrt();
    let dvar_y = hadamard_mean(&bc, &bc).sqrt();
    let dcor = dcov / (dvar_x * dvar_y).sqrt();
    let test_statistic = n as f64 * dcov * dcov;

    Ok(DistDependStat {
        test_statistic,
        distance_correlation: dcor,
        distance_covariance: dcov,
        dvar_x,
        dvar_y,
        s,
    })
}

/// Empirical distance covariance `dCov(X, Y)`.
pub fn distance_covariance(x: &Array2<f64>, y: &Array2<f64>) -> Result<f64> {
    Ok(distance_statistics(x, y)?.distance_covariance)
}

/// Empirical distance correlation `dCor(X, Y)` in `[0, 1]`.
pub fn distance_correlation(x: &Array2<f64>, y: &Array2<f64>) -> Result<f64> {
    Ok(distance_statistics(x, y)?.distance_correlation)
}

/// Empirical distance variance of `X`, `dVar(X) = dCov(X, X)`.
pub fn distance_variance(x: &Array2<f64>) -> Result<f64> {
    Ok(distance_statistics(x, x)?.distance_covariance)
}

/// Asymptotic distance-covariance (dCov) test of independence.
///
/// Returns the statistic `sqrt(n · dCov² / S)` and its two-sided standard-normal
/// p-value (the reference `_asymptotic_pvalue`).
pub fn distance_covariance_test(x: &Array2<f64>, y: &Array2<f64>) -> Result<DcovTest> {
    let stats = distance_statistics(x, y)?;
    let statistic = (stats.test_statistic / stats.s).sqrt();
    let pvalue = (1.0 - norm_cdf(statistic)) * 2.0;
    Ok(DcovTest { statistic, pvalue })
}

/// Reshape a 1-D sample into an `n × 1` matrix (the column-vector form expected
/// by [`distance_statistics`]).
pub fn as_column(v: &Array1<f64>) -> Array2<f64> {
    let n = v.len();
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        out[[i, 0]] = v[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dcor_of_identical_is_one() {
        let x = array![1.0, 2.0, 3.0, 5.0, 8.0];
        let xc = as_column(&x);
        let st = distance_statistics(&xc, &xc).unwrap();
        assert!((st.distance_correlation - 1.0).abs() < 1e-12);
        // dVar_x == dVar_y == dCov when x == y.
        assert!((st.dvar_x - st.distance_covariance).abs() < 1e-12);
        assert!((st.dvar_y - st.distance_covariance).abs() < 1e-12);
    }

    #[test]
    fn dcor_in_unit_interval() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 0.5, 2.2, -1.0, 3.3, 0.1];
        let st = distance_statistics(&as_column(&x), &as_column(&y)).unwrap();
        assert!((0.0..=1.0).contains(&st.distance_correlation));
        assert!(st.distance_covariance >= 0.0);
    }

    #[test]
    fn mismatched_lengths_error() {
        let x = as_column(&array![1.0, 2.0, 3.0]);
        let y = as_column(&array![1.0, 2.0]);
        assert!(distance_statistics(&x, &y).is_err());
    }
}
