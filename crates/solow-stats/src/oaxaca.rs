//! Blinder-Oaxaca decomposition of the mean gap between two groups.
//!
//! [`OaxacaBlinder`] decomposes the difference in the mean of an outcome between
//! two groups (defined by a binary column of the design matrix) into components
//! attributable to differences in covariate *endowments* and differences in the
//! estimated *coefficients*. Both the classic three-fold decomposition
//! (endowments / coefficients / interaction) and the two-fold "pooled"
//! decomposition (explained / unexplained) are provided, mirroring the reference
//! `stats.oaxaca.OaxacaBlinder`.
//!
//! Given two groups `f` (first) and `s` (second) with covariate means
//! `x̄_f`, `x̄_s` and fitted OLS coefficients `β_f`, `β_s`, the three-fold
//! decomposition writes the gap `ȳ_f − ȳ_s` as
//!
//! ```text
//! endowments   = (x̄_f − x̄_s) · β_s
//! coefficients =  x̄_s · (β_f − β_s)
//! interaction  = (x̄_f − x̄_s) · (β_f − β_s)
//! ```
//!
//! and the two-fold decomposition (with a non-discriminatory coefficient vector
//! `β*`) as
//!
//! ```text
//! explained   = (x̄_f − x̄_s) · β*
//! unexplained =  x̄_f · (β_f − β*) + x̄_s · (β* − β_s)
//! ```
//!
//! Closed-form OLS underlies every quantity, so the effects reproduce the
//! reference to machine precision.

use ndarray::{Array1, Array2};
use solow_core::{Error, Result};
use solow_regression::LinearModel;

/// Weighting scheme for the non-discriminatory coefficient vector `β*` used by
/// the two-fold ("pooled") decomposition. Mirrors the reference
/// `two_fold_type` options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TwoFoldType {
    /// `β*` from OLS on the full sample including the group indicator; the
    /// indicator's coefficient is dropped before applying `β*`.
    Pooled,
    /// `β*` from OLS on the full sample *excluding* the group indicator
    /// (the "Neumark" pooled model).
    Neumark,
    /// Cotton (1988) sample-size weighting:
    /// `β* = (n_f β_f + n_s β_s) / (n_f + n_s)`.
    Cotton,
    /// Reimers 50/50 weighting: `β* = ½ (β_f + β_s)`.
    Reimers,
    /// User-supplied weight `w` on the larger-mean group:
    /// `β* = w β_f + (1 − w) β_s`.
    SelfSubmitted(f64),
}

/// Two-fold ("pooled") Blinder-Oaxaca decomposition of the mean gap.
#[derive(Debug, Clone)]
pub struct TwoFold {
    /// Effect attributable to differences in coefficients (and the constant),
    /// i.e. the part *not* explained by the covariate endowments.
    pub unexplained: f64,
    /// Effect attributable to differences in covariate endowments.
    pub explained: f64,
    /// The raw mean gap `ȳ_f − ȳ_s` (non-negative after the group swap).
    pub gap: f64,
}

/// Three-fold Blinder-Oaxaca decomposition of the mean gap.
#[derive(Debug, Clone)]
pub struct ThreeFold {
    /// Endowments (characteristics) effect `(x̄_f − x̄_s) · β_s`.
    pub endowments: f64,
    /// Coefficients effect `x̄_s · (β_f − β_s)`.
    pub coefficients: f64,
    /// Interaction effect `(x̄_f − x̄_s) · (β_f − β_s)`.
    pub interaction: f64,
    /// The raw mean gap `ȳ_f − ȳ_s`.
    pub gap: f64,
}

/// Blinder-Oaxaca decomposition model.
///
/// `endog` is the outcome, `exog` the design matrix, and `bifurcate` the column
/// index of the binary group indicator. With `hasconst = true` the design is
/// assumed to already contain a constant; with `hasconst = false` a constant is
/// appended to each group's covariate matrix (matching the reference, which
/// appends rather than prepends). The two groups are ordered so the first group
/// has the larger outcome mean (the reference `swap=True` behaviour), making the
/// reported gap non-negative.
#[derive(Debug, Clone)]
pub struct OaxacaBlinder {
    bifurcate: usize,
    hasconst: bool,
    /// Design with the bifurcate column removed (used for the Neumark pooled fit).
    neumark: Array2<f64>,
    /// Full design (used for the pooled fit).
    exog: Array2<f64>,
    /// Full outcome vector.
    endog: Array1<f64>,
    gap: f64,
    len_f: usize,
    len_s: usize,
    exog_f_mean: Array1<f64>,
    exog_s_mean: Array1<f64>,
    f_params: Array1<f64>,
    s_params: Array1<f64>,
}

/// Append a constant column of ones at the end of `x` (`prepend=False`).
fn add_constant_append(x: &Array2<f64>) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut out = Array2::<f64>::zeros((n, k + 1));
    out.slice_mut(ndarray::s![.., ..k]).assign(x);
    for i in 0..n {
        out[[i, k]] = 1.0;
    }
    out
}

/// Mean of each column of `x`.
fn col_means(x: &Array2<f64>) -> Array1<f64> {
    let (n, k) = x.dim();
    let mut m = Array1::<f64>::zeros(k);
    for j in 0..k {
        let mut s = 0.0;
        for i in 0..n {
            s += x[[i, j]];
        }
        m[j] = s / n as f64;
    }
    m
}

/// Drop column `col` from `x`.
fn delete_col(x: &Array2<f64>, col: usize) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut out = Array2::<f64>::zeros((n, k - 1));
    let mut jj = 0;
    for j in 0..k {
        if j == col {
            continue;
        }
        for i in 0..n {
            out[[i, jj]] = x[[i, j]];
        }
        jj += 1;
    }
    out
}

/// Select the rows of `x` whose index appears in `rows`.
fn select_rows(x: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    let k = x.ncols();
    let mut out = Array2::<f64>::zeros((rows.len(), k));
    for (ii, &i) in rows.iter().enumerate() {
        for j in 0..k {
            out[[ii, j]] = x[[i, j]];
        }
    }
    out
}

fn select_elems(v: &Array1<f64>, rows: &[usize]) -> Array1<f64> {
    Array1::from_iter(rows.iter().map(|&i| v[i]))
}

fn mean(v: &Array1<f64>) -> f64 {
    v.sum() / v.len() as f64
}

fn fit_params(endog: Array1<f64>, exog: Array2<f64>) -> Result<Array1<f64>> {
    let res = LinearModel::ols(endog, exog)?.fit()?;
    Ok(res.params)
}

impl OaxacaBlinder {
    /// Build the decomposition model.
    ///
    /// `bifurcate` is the column index of the binary group indicator in `exog`.
    /// The indicator must take exactly two distinct values.
    pub fn new(
        endog: Array1<f64>,
        exog: Array2<f64>,
        bifurcate: usize,
        hasconst: bool,
    ) -> Result<Self> {
        let n = endog.len();
        if exog.nrows() != n {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if bifurcate >= exog.ncols() {
            return Err(Error::Value("bifurcate column out of range".into()));
        }

        // Unique group values, ascending (np.unique). Require exactly two.
        let bi_col: Vec<f64> = (0..n).map(|i| exog[[i, bifurcate]]).collect();
        let mut uniq: Vec<f64> = bi_col.clone();
        uniq.sort_by(|a, b| a.total_cmp(b));
        uniq.dedup();
        if uniq.len() != 2 {
            return Err(Error::Value(
                "bifurcate column must take exactly two distinct values".into(),
            ));
        }
        let mut bi = [uniq[0], uniq[1]];

        // Row index sets for the two groups.
        let mut rows_f: Vec<usize> = (0..n).filter(|&i| bi_col[i] == bi[0]).collect();
        let mut rows_s: Vec<usize> = (0..n).filter(|&i| bi_col[i] == bi[1]).collect();

        let endog_full = endog.clone();
        let mut endog_f = select_elems(&endog_full, &rows_f);
        let mut endog_s = select_elems(&endog_full, &rows_s);

        // The reference fixes `len_f`/`len_s` from the *initial* group ordering
        // (before any swap) and reuses them in the Cotton weighting, so we
        // capture them here, prior to the swap below.
        let len_f = rows_f.len();
        let len_s = rows_s.len();

        let mut gap = mean(&endog_f) - mean(&endog_s);

        // swap=True (reference default): order so the first group has the larger
        // outcome mean, making the gap non-negative.
        if gap < 0.0 {
            std::mem::swap(&mut rows_f, &mut rows_s);
            std::mem::swap(&mut endog_f, &mut endog_s);
            bi.swap(0, 1);
            gap = mean(&endog_f) - mean(&endog_s);
        }

        // Group covariate matrices: rows of the group, bifurcate column deleted.
        let mut exog_f = delete_col(&select_rows(&exog, &rows_f), bifurcate);
        let mut exog_s = delete_col(&select_rows(&exog, &rows_s), bifurcate);

        let neumark = delete_col(&exog, bifurcate);
        let (exog_full, neumark) = if hasconst {
            (exog.clone(), neumark)
        } else {
            exog_f = add_constant_append(&exog_f);
            exog_s = add_constant_append(&exog_s);
            (add_constant_append(&exog), add_constant_append(&neumark))
        };

        let exog_f_mean = col_means(&exog_f);
        let exog_s_mean = col_means(&exog_s);

        let f_params = fit_params(endog_f, exog_f)?;
        let s_params = fit_params(endog_s, exog_s)?;

        Ok(OaxacaBlinder {
            bifurcate,
            hasconst,
            neumark,
            exog: exog_full,
            endog,
            gap,
            len_f,
            len_s,
            exog_f_mean,
            exog_s_mean,
            f_params,
            s_params,
        })
    }

    /// The non-negative mean gap `ȳ_f − ȳ_s`.
    pub fn gap(&self) -> f64 {
        self.gap
    }

    /// Fitted first-group coefficients `β_f`.
    pub fn f_params(&self) -> &Array1<f64> {
        &self.f_params
    }

    /// Fitted second-group coefficients `β_s`.
    pub fn s_params(&self) -> &Array1<f64> {
        &self.s_params
    }

    /// First-group covariate means `x̄_f`.
    pub fn exog_f_mean(&self) -> &Array1<f64> {
        &self.exog_f_mean
    }

    /// Second-group covariate means `x̄_s`.
    pub fn exog_s_mean(&self) -> &Array1<f64> {
        &self.exog_s_mean
    }

    /// Non-discriminatory coefficient vector `β*` for the given weighting.
    fn t_params(&self, kind: TwoFoldType) -> Result<Array1<f64>> {
        Ok(match kind {
            TwoFoldType::Pooled => {
                let full = fit_params(self.endog.clone(), self.exog.clone())?;
                // Drop the bifurcate coefficient.
                Array1::from_iter(
                    (0..full.len())
                        .filter(|&j| j != self.bifurcate)
                        .map(|j| full[j]),
                )
            }
            TwoFoldType::Neumark => fit_params(self.endog.clone(), self.neumark.clone())?,
            TwoFoldType::Cotton => {
                let nf = self.len_f as f64;
                let ns = self.len_s as f64;
                &self.f_params * (nf / (nf + ns)) + &self.s_params * (ns / (nf + ns))
            }
            TwoFoldType::Reimers => (&self.f_params + &self.s_params) * 0.5,
            TwoFoldType::SelfSubmitted(w) => &self.f_params * w + &self.s_params * (1.0 - w),
        })
    }

    /// Two-fold ("pooled") decomposition with the requested weighting scheme.
    pub fn two_fold(&self, kind: TwoFoldType) -> Result<TwoFold> {
        let tp = self.t_params(kind)?;
        if tp.len() != self.f_params.len() {
            return Err(Error::Shape("t_params dimension mismatch".into()));
        }
        let unexplained = self.exog_f_mean.dot(&(&self.f_params - &tp))
            + self.exog_s_mean.dot(&(&tp - &self.s_params));
        let explained = (&self.exog_f_mean - &self.exog_s_mean).dot(&tp);
        Ok(TwoFold {
            unexplained,
            explained,
            gap: self.gap,
        })
    }

    /// Three-fold decomposition (endowments / coefficients / interaction).
    pub fn three_fold(&self) -> ThreeFold {
        let dmean = &self.exog_f_mean - &self.exog_s_mean;
        let dparams = &self.f_params - &self.s_params;
        ThreeFold {
            endowments: dmean.dot(&self.s_params),
            coefficients: self.exog_s_mean.dot(&dparams),
            interaction: dmean.dot(&dparams),
            gap: self.gap,
        }
    }

    /// Whether the design was supplied with a constant (`hasconst`).
    pub fn hasconst(&self) -> bool {
        self.hasconst
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn toy() -> (Array1<f64>, Array2<f64>) {
        // [const, group, x]
        let exog = array![
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 1.0, 3.0],
        ];
        let endog = array![1.0, 2.0, 3.0, 4.0, 5.5, 7.0];
        (endog, exog)
    }

    #[test]
    fn gap_is_nonnegative_and_decompositions_sum_to_gap() {
        let (y, x) = toy();
        let m = OaxacaBlinder::new(y, x, 1, true).unwrap();
        assert!(m.gap() >= 0.0);

        let tf = m.three_fold();
        let s = tf.endowments + tf.coefficients + tf.interaction;
        assert!((s - tf.gap).abs() < 1e-9, "three-fold sums to gap");

        let two = m.two_fold(TwoFoldType::Pooled).unwrap();
        assert!((two.unexplained + two.explained - two.gap).abs() < 1e-9);
    }

    #[test]
    fn rejects_non_binary_group() {
        let exog = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let endog = array![1.0, 2.0, 3.0];
        assert!(OaxacaBlinder::new(endog, exog, 1, true).is_err());
    }
}
