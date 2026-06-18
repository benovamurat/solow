//! Analysis-of-variance tables for a fitted linear model (types I, II, III).
//!
//! Mirrors the reference `anova_lm` for a single model. Because this crate has
//! no formula parser, the caller supplies the model in *term* form: the design
//! matrix `exog`, the response `endog`, and a list of named terms each mapping
//! to a contiguous column range of `exog` (a categorical factor with `m` levels
//! occupies `m − 1` columns, exactly as the reference's `design_info`). The
//! intercept term, if present, must be the first term and span column 0.
//!
//! - **Type I** (sequential) uses the QR "effects" of the design: each term's
//!   sum of squares is the squared length of the effects in its columns.
//! - **Type II** (marginal, hierarchy-respecting) and **Type III** (marginal,
//!   each term adjusted for all others) form a linear restriction `L` for each
//!   term and back the sum of squares out of the general linear F-test.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::f_sf;
use solow_linalg::{pinv, qr};

/// Type of sum of squares for [`anova_lm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnovaType {
    /// Sequential (Type I).
    I,
    /// Marginal, hierarchy-respecting (Type II).
    II,
    /// Marginal, fully adjusted (Type III).
    III,
}

/// A named model term and the half-open column range `[start, stop)` it spans
/// in the design matrix.
#[derive(Debug, Clone)]
pub struct Term {
    /// Term label, e.g. `"C(a)"` or `"C(a):C(b)"`. The interaction separator is
    /// `:`; factor names are the colon-separated pieces.
    pub name: String,
    /// First column (inclusive) of this term in `exog`.
    pub start: usize,
    /// Last column (exclusive) of this term in `exog`.
    pub stop: usize,
}

impl Term {
    /// Construct a term spanning columns `[start, stop)`.
    pub fn new(name: impl Into<String>, start: usize, stop: usize) -> Self {
        Term {
            name: name.into(),
            start,
            stop,
        }
    }

    /// Set of constituent factor names (split on the interaction separator).
    fn factors(&self) -> Vec<&str> {
        self.name.split(':').collect()
    }
}

/// One row of an ANOVA table.
#[derive(Debug, Clone)]
pub struct AnovaRow {
    /// Row label (a term name, or `"Residual"`).
    pub name: String,
    /// Degrees of freedom.
    pub df: f64,
    /// Sum of squares.
    pub sum_sq: f64,
    /// Mean square `sum_sq / df`.
    pub mean_sq: f64,
    /// F statistic (`None` for the residual row).
    pub f: Option<f64>,
    /// p-value `PR(>F)` (`None` for the residual row).
    pub pr: Option<f64>,
}

/// A full ANOVA table.
#[derive(Debug, Clone)]
pub struct AnovaTable {
    /// Table rows, in display order (terms first, then `Residual`).
    pub rows: Vec<AnovaRow>,
}

impl AnovaTable {
    /// Look up a row by its label.
    pub fn row(&self, name: &str) -> Option<&AnovaRow> {
        self.rows.iter().find(|r| r.name == name)
    }
}

/// Whether `name` denotes the intercept term.
fn is_intercept(name: &str) -> bool {
    name == "Intercept"
}

/// ANOVA table for one fitted linear model. Mirrors the single-model
/// `anova_lm(model, typ=...)`.
///
/// `exog`/`endog` are the design and response, `terms` the named column groups,
/// and `typ` the sum-of-squares type. The model is refit internally by ordinary
/// least squares (pseudo-inverse), matching the reference's OLS fit.
pub fn anova_lm(
    endog: &Array1<f64>,
    exog: &Array2<f64>,
    terms: &[Term],
    typ: AnovaType,
) -> Result<AnovaTable> {
    let (nobs, kcols) = exog.dim();

    // OLS fit via the pseudo-inverse (pinv returns (A^+, singular values)).
    let (pinv_x, sv) = pinv(exog)?;
    let beta = pinv_x.dot(endog);
    let fitted = exog.dot(&beta);
    let resid = endog - &fitted;
    let ssr = resid.dot(&resid);

    // Rank and residual degrees of freedom (reference matrix_rank convention).
    let smax = sv.iter().cloned().fold(0.0_f64, f64::max);
    let tol = smax * (sv.len() as f64) * f64::EPSILON;
    let rank = sv.iter().filter(|&&s| s > tol).count();
    let df_resid = nobs as f64 - rank as f64;
    let scale = ssr / df_resid;

    let rows = match typ {
        AnovaType::I => anova_type1(endog, exog, terms, ssr, df_resid),
        AnovaType::II | AnovaType::III => {
            // Parameter covariance: scale · (XᵀX)^+.
            let xtx = exog.t().dot(exog);
            let (xtx_pinv, _) = pinv(&xtx)?;
            let cov = &xtx_pinv * scale;
            if typ == AnovaType::III {
                anova_type3(&beta, &cov, terms, kcols, ssr, df_resid)?
            } else {
                anova_type2(&beta, &cov, terms, kcols, ssr, df_resid)?
            }
        }
    };

    Ok(AnovaTable { rows })
}

/// Assemble a finished row given its sum of squares and degrees of freedom.
fn make_row(name: &str, sum_sq: f64, df: f64, scale: f64, df_resid: f64) -> AnovaRow {
    let mean_sq = sum_sq / df;
    let f = mean_sq / scale;
    let pr = f_sf(f, df, df_resid);
    AnovaRow {
        name: name.to_string(),
        df,
        sum_sq,
        mean_sq,
        f: Some(f),
        pr: Some(pr),
    }
}

/// Residual row (no F / p-value).
fn residual_row(ssr: f64, df_resid: f64) -> AnovaRow {
    AnovaRow {
        name: "Residual".to_string(),
        df: df_resid,
        sum_sq: ssr,
        mean_sq: ssr / df_resid,
        f: None,
        pr: None,
    }
}

/// Type I (sequential) sums of squares via the QR effects.
fn anova_type1(
    endog: &Array1<f64>,
    exog: &Array2<f64>,
    terms: &[Term],
    ssr: f64,
    df_resid: f64,
) -> Vec<AnovaRow> {
    let scale = ssr / df_resid;
    // effects = Qᵀ y from the reduced QR of the design.
    let (q, _r) = qr(exog).expect("qr of design");
    let effects = q.t().dot(endog);

    let mut rows = Vec::new();
    for t in terms {
        if is_intercept(&t.name) {
            continue;
        }
        let mut ss = 0.0;
        for c in t.start..t.stop {
            ss += effects[c] * effects[c];
        }
        let df = (t.stop - t.start) as f64;
        rows.push(make_row(&t.name, ss, df, scale, df_resid));
    }
    rows.push(residual_row(ssr, df_resid));
    rows
}

/// Type III sums of squares: each term tested as `term == 0` adjusting for all
/// others. The restriction `L` is the identity rows of the term's columns.
fn anova_type3(
    beta: &Array1<f64>,
    cov: &Array2<f64>,
    terms: &[Term],
    kcols: usize,
    ssr: f64,
    df_resid: f64,
) -> Result<Vec<AnovaRow>> {
    let scale = ssr / df_resid;
    let mut rows = Vec::new();
    for t in terms {
        let l = identity_rows(kcols, &(t.start..t.stop).collect::<Vec<_>>());
        let (fstat, r) = f_test(&l, beta, cov)?;
        let df = r as f64;
        let ss = fstat * df * scale;
        rows.push(make_row(&t.name, ss, df, scale, df_resid));
    }
    rows.push(residual_row(ssr, df_resid));
    Ok(rows)
}

/// Type II sums of squares: each term tested against the model that contains
/// all terms not marginal-to it, using the orthogonal-complement restriction.
fn anova_type2(
    beta: &Array1<f64>,
    cov: &Array2<f64>,
    terms: &[Term],
    kcols: usize,
    ssr: f64,
    df_resid: f64,
) -> Result<Vec<AnovaRow>> {
    let scale = ssr / df_resid;
    // Terms excluding the intercept.
    let model_terms: Vec<&Term> = terms.iter().filter(|t| !is_intercept(&t.name)).collect();

    let mut rows = Vec::new();
    for term in &model_terms {
        let mut l1_cols: Vec<usize> = (term.start..term.stop).collect();
        let mut l2_cols: Vec<usize> = Vec::new();
        let term_set: Vec<&str> = term.factors();
        for t in &model_terms {
            let other: Vec<&str> = t.factors();
            // term is a strict subset of t (higher-order term containing it).
            if is_strict_subset(&term_set, &other) {
                l1_cols.extend(t.start..t.stop);
                l2_cols.extend(t.start..t.stop);
            }
        }
        let l1 = identity_rows(kcols, &l1_cols);
        let (l12, r) = if !l2_cols.is_empty() {
            let l2 = identity_rows(kcols, &l2_cols);
            // LVL = L1 cov L2ᵀ ; take the last r columns of the full QR of LVL.
            let lvl = l1.dot(cov).dot(&l2.t());
            let rr = l1.nrows() - l2.nrows();
            let orth_compl = full_q(&lvl)?;
            let ncolq = orth_compl.ncols();
            let comp = orth_compl.slice(ndarray::s![.., (ncolq - rr)..]).to_owned();
            (comp.t().dot(&l1), rr)
        } else {
            (l1.clone(), l1.nrows())
        };

        let (fstat, _jr) = f_test(&l12, beta, cov)?;
        let df = r as f64;
        let ss = fstat * df * scale;
        rows.push(make_row(&term.name, ss, df, scale, df_resid));
    }
    rows.push(residual_row(ssr, df_resid));
    Ok(rows)
}

/// `a` is a strict subset of `b` (set semantics over factor names).
fn is_strict_subset(a: &[&str], b: &[&str]) -> bool {
    if a.len() >= b.len() {
        return false;
    }
    a.iter().all(|x| b.contains(x))
}

/// Build the restriction matrix `L` of identity rows for `cols`, shape
/// `cols.len() × kcols`.
fn identity_rows(kcols: usize, cols: &[usize]) -> Array2<f64> {
    let mut l = Array2::<f64>::zeros((cols.len(), kcols));
    for (i, &c) in cols.iter().enumerate() {
        l[[i, c]] = 1.0;
    }
    l
}

/// General linear F-test for the restriction `L b = 0`.
///
/// Returns `(F, J)` where `J = rank(L)` rows and
/// `F = (Lb)ᵀ (L cov Lᵀ)^+ (Lb) / J`. Mirrors the reference `f_test`.
fn f_test(l: &Array2<f64>, beta: &Array1<f64>, cov: &Array2<f64>) -> Result<(f64, usize)> {
    let rb = l.dot(beta);
    let cov_l = l.dot(cov).dot(&l.t());
    let (cov_l_pinv, _) = pinv(&cov_l)?;
    let quad = rb.dot(&cov_l_pinv.dot(&rb));
    let j = l.nrows();
    Ok((quad / j as f64, j))
}

/// Full (square) `Q` of the QR decomposition of `a` (shape `m × n`, `m ≥ n`),
/// returned as an `m × m` orthogonal matrix. Implemented with local Householder
/// reflectors so the orthogonal complement (last `m − rank` columns) is
/// available, which the economy QR does not expose.
fn full_q(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, n) = a.dim();
    if m < n {
        return Err(Error::Shape("full_q requires rows >= cols".into()));
    }
    let mut r = a.clone();
    let mut q = Array2::<f64>::eye(m);
    for k in 0..n {
        let mut norm = 0.0;
        for i in k..m {
            norm += r[[i, k]] * r[[i, k]];
        }
        let norm = norm.sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if r[[k, k]] >= 0.0 { -norm } else { norm };
        let mut v = vec![0.0; m];
        v[k] = r[[k, k]] - alpha;
        for (i, vi) in v.iter_mut().enumerate().skip(k + 1) {
            *vi = r[[i, k]];
        }
        let mut vnorm2 = 0.0;
        for vi in v.iter().skip(k) {
            vnorm2 += vi * vi;
        }
        if vnorm2 == 0.0 {
            continue;
        }
        // Apply H to R.
        for j in k..n {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i] * r[[i, j]];
            }
            let b = 2.0 * dot / vnorm2;
            for i in k..m {
                r[[i, j]] -= b * v[i];
            }
        }
        // Apply H to all columns of Q.
        for j in 0..m {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i] * q[[j, i]];
            }
            let b = 2.0 * dot / vnorm2;
            for i in k..m {
                q[[j, i]] -= b * v[i];
            }
        }
    }
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn simple_regression_type1() {
        // y = 1 + 2 x exactly: one term, residual SS = 0.
        let exog = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let endog = array![1.0, 3.0, 5.0, 7.0];
        let terms = vec![Term::new("Intercept", 0, 1), Term::new("x", 1, 2)];
        let tab = anova_lm(&endog, &exog, &terms, AnovaType::I).unwrap();
        let x = tab.row("x").unwrap();
        assert!(x.sum_sq > 0.0);
        let res = tab.row("Residual").unwrap();
        assert!(res.sum_sq.abs() < 1e-18);
    }

    #[test]
    fn strict_subset_logic() {
        assert!(is_strict_subset(&["a"], &["a", "b"]));
        assert!(!is_strict_subset(&["a", "b"], &["a", "b"]));
        assert!(!is_strict_subset(&["c"], &["a", "b"]));
    }
}
